"""Flask webapp: single-user login + dashboard for the arXiv daily digest.

Run with:
    python -m webapp.app

Required environment variables: see .env.example.

The pipeline runs in a background thread and streams progress events to the
browser via Server-Sent Events. The browser opens /progress, which connects to
/events and renders incoming JSON events as a live log.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import secrets
import threading
import time
import traceback
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from flask import (
    Flask,
    Response,
    flash,
    redirect,
    render_template,
    request,
    session,
    stream_with_context,
    url_for,
)
from werkzeug.security import check_password_hash

from src.ingestion.storage import PaperStore
from src.pipeline import Digest, run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_PROFILE = ROOT / "config" / "user_profile.example.json"
# If PROFILE_PATH is not set, use config/my_profile.json. Auto-copy from
# the example on first read so the shipped example file stays pristine
# and isn't mutated when the user edits via the UI.
DEFAULT_EDITABLE_PROFILE = ROOT / "config" / "my_profile.json"
PROFILE_PATH = Path(os.environ.get("PROFILE_PATH", DEFAULT_EDITABLE_PROFILE))
DB_PATH = Path(os.environ.get("DB_PATH", ROOT / "data" / "papers.db"))
LATEST_DIGEST_PATH = ROOT / "data" / "latest_digest.json"


def _ensure_profile_exists() -> None:
    """Copy the example profile into place if the editable profile is missing."""
    if not PROFILE_PATH.exists() and EXAMPLE_PROFILE.exists():
        PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        PROFILE_PATH.write_text(EXAMPLE_PROFILE.read_text())
        logger.info("Bootstrapped %s from example profile", PROFILE_PATH)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32)


# --- Pipeline run state -------------------------------------------------------
#
# Single-user app, so a module-level state object is fine. Each pipeline run
# pushes events onto a queue; the SSE endpoint reads from the queue and yields
# them to the browser. We also keep a list of past events for the current run
# so a page reload doesn't lose history.


class RunState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.queue: queue.Queue = queue.Queue()
        self.events: list[dict] = []
        self.status: str = "idle"  # idle | running | done | error
        self.started_at: float | None = None
        self.error: str | None = None
        self.no_new_papers: bool = False

    def reset(self) -> None:
        with self.lock:
            self.queue = queue.Queue()
            self.events = []
            self.status = "running"
            self.started_at = time.time()
            self.error = None
            self.no_new_papers = False

    def emit(self, event_type: str, message: str, **data) -> None:
        evt = {
            "type": event_type,
            "message": message,
            "timestamp": time.time(),
            **data,
        }
        with self.lock:
            self.events.append(evt)
        self.queue.put(evt)

    def finish(self, status: str, error: str | None = None) -> None:
        with self.lock:
            self.status = status
            self.error = error
        # Sentinel so the SSE loop can shut down
        self.queue.put(None)


run_state = RunState()


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)

    return wrapped


def _load_latest_digest() -> tuple[dict | None, str | None]:
    if not LATEST_DIGEST_PATH.exists():
        return None, None
    raw = json.loads(LATEST_DIGEST_PATH.read_text())
    return raw.get("digest"), raw.get("generated_at")


def _save_latest_digest(digest: Digest) -> str:
    LATEST_DIGEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()
    LATEST_DIGEST_PATH.write_text(
        json.dumps(
            {"generated_at": generated_at, "digest": digest.to_dict()},
            indent=2,
        )
    )
    return generated_at


# --- Auth ---------------------------------------------------------------------


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        expected_user = os.environ.get("APP_USERNAME")
        expected_hash = os.environ.get("APP_PASSWORD_HASH")
        if not expected_user or not expected_hash:
            flash("Server not configured: APP_USERNAME / APP_PASSWORD_HASH missing.", "error")
            return render_template("login.html"), 500
        if username == expected_user and check_password_hash(expected_hash, password):
            session["logged_in"] = True
            session["username"] = username
            return redirect(request.args.get("next") or url_for("dashboard"))
        flash("Invalid credentials.", "error")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# --- Pages --------------------------------------------------------------------


def _last_run_info(generated_at: str | None) -> tuple[int | None, int]:
    """Return (days_since_last_run, recommended_lookback).

    days_since_last_run is None if there's never been a run. The recommended
    lookback is clamped between 1 and the input field's max (14). The point is
    to hint to the user how wide to set their fetch window when catching up.
    """
    default_recommended = 2
    if not generated_at:
        return None, default_recommended
    try:
        last = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    except ValueError:
        return None, default_recommended
    now = datetime.now(timezone.utc)
    delta = now - last
    days_since = max(0, delta.days)
    # Recommend lookback = days_since + 1 (so we cover the day they last ran
    # AND today) — clamped to the input's max of 14.
    recommended = max(1, min(14, days_since + 1))
    return days_since, recommended


@app.route("/")
@login_required
def dashboard():
    digest, generated_at = _load_latest_digest()
    days_since_last_run, recommended_lookback = _last_run_info(generated_at)
    # Consume one-shot "no new papers" notice from the last run, if any
    no_new_papers = False
    with run_state.lock:
        if run_state.no_new_papers:
            no_new_papers = True
            run_state.no_new_papers = False
    bookmarked = PaperStore(DB_PATH).bookmarked_ids() if DB_PATH.exists() else set()
    return render_template(
        "digest.html",
        digest=digest,
        generated_at=generated_at,
        days_since_last_run=days_since_last_run,
        recommended_lookback=recommended_lookback,
        no_new_papers=no_new_papers,
        bookmarked_ids=bookmarked,
        username=session.get("username"),
    )


@app.route("/bookmark/<arxiv_id>", methods=["POST"])
@login_required
def toggle_bookmark(arxiv_id: str):
    is_bookmarked = PaperStore(DB_PATH).toggle_bookmark(arxiv_id)
    flash(
        f"Bookmarked {arxiv_id}." if is_bookmarked else f"Removed bookmark for {arxiv_id}.",
        "success",
    )
    # Send the user back to whichever page they clicked from.
    return redirect(request.referrer or url_for("dashboard"))


@app.route("/bookmarks")
@login_required
def bookmarks():
    store = PaperStore(DB_PATH)
    entries = store.load_bookmarked()
    return render_template(
        "bookmarks.html",
        entries=entries,
        username=session.get("username"),
    )


@app.route("/progress")
@login_required
def progress():
    return render_template("progress.html", username=session.get("username"))


# --- Pipeline trigger + SSE ---------------------------------------------------


def _run_pipeline_in_background(lookback: int, threshold: float, shortlist_max: int) -> None:
    """Run the pipeline, pushing events into run_state.queue."""
    try:
        digest = run_pipeline(
            profile_path=PROFILE_PATH,
            db_path=DB_PATH,
            lookback_days=lookback,
            shortlist_threshold=threshold,
            shortlist_max=shortlist_max,
            on_progress=lambda evt_type, message, **data: run_state.emit(
                evt_type, message, **data
            ),
        )
        # Only overwrite the cached digest if we actually produced new content.
        # An empty run (e.g. all fetched papers were already in dedup) shouldn't
        # blow away a previously-cached good digest.
        if digest.new_papers > 0 or digest.shortlist:
            _save_latest_digest(digest)
            run_state.no_new_papers = False
        else:
            run_state.no_new_papers = True
        run_state.finish("done")
    except Exception as e:
        logger.exception("Pipeline failed")
        tb = traceback.format_exc()[-600:]
        run_state.emit("error", f"Pipeline failed: {e}", traceback=tb)
        run_state.finish("error", error=str(e))


@app.route("/refresh", methods=["POST"])
@login_required
def refresh():
    if run_state.status == "running":
        flash("A pipeline run is already in progress.", "error")
        return redirect(url_for("progress"))

    if not PROFILE_PATH.exists():
        flash(f"Profile not found at {PROFILE_PATH}", "error")
        return redirect(url_for("dashboard"))

    lookback = int(request.form.get("lookback", "2"))
    threshold = float(request.form.get("threshold", "0.6"))
    shortlist_max = int(request.form.get("shortlist_max", "10"))

    run_state.reset()
    thread = threading.Thread(
        target=_run_pipeline_in_background,
        args=(lookback, threshold, shortlist_max),
        daemon=True,
    )
    thread.start()
    return redirect(url_for("progress"))


@app.route("/events")
@login_required
def events():
    """Server-Sent Events stream for the in-flight pipeline run.

    Yields all events the run has emitted so far (so a page reload can catch
    up), then blocks on the queue for new events until the run completes.
    """

    def stream():
        # Replay history first (for late-joining clients / reloads)
        with run_state.lock:
            history = list(run_state.events)
            status = run_state.status

        for evt in history:
            yield f"event: progress\ndata: {json.dumps(evt)}\n\n"

        if status in ("done", "error", "idle"):
            yield f"event: {status}\ndata: {{}}\n\n"
            return

        # Tail the live queue
        while True:
            try:
                evt = run_state.queue.get(timeout=30)
            except queue.Empty:
                # Keep the connection alive
                yield ": keepalive\n\n"
                continue
            if evt is None:
                # Pipeline signalled completion
                with run_state.lock:
                    final = run_state.status
                yield f"event: {final}\ndata: {{}}\n\n"
                return
            yield f"event: progress\ndata: {json.dumps(evt)}\n\n"

    return Response(
        stream_with_context(stream()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering if behind a proxy
        },
    )


@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    """Edit the user's research interest profile."""
    _ensure_profile_exists()

    if request.method == "POST":
        try:
            existing = json.loads(PROFILE_PATH.read_text()) if PROFILE_PATH.exists() else {}
        except json.JSONDecodeError:
            existing = {}

        def _lines(field: str) -> list[str]:
            raw = request.form.get(field, "")
            return [line.strip() for line in raw.splitlines() if line.strip()]

        def _csv(field: str) -> list[str]:
            raw = request.form.get(field, "")
            return [item.strip() for item in raw.split(",") if item.strip()]

        new_profile = {
            "name": request.form.get("name", "").strip() or "Researcher",
            "research_interests": _lines("research_interests"),
            "methodological_preferences": _lines("methodological_preferences"),
            "avoid": _lines("avoid"),
            "arxiv_categories": _csv("arxiv_categories"),
            "keywords": _csv("keywords"),
            # Preserve rating history — UI doesn't expose it yet.
            "rating_history": existing.get("rating_history", []),
        }
        PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        PROFILE_PATH.write_text(json.dumps(new_profile, indent=2))
        flash(f"Profile saved to {PROFILE_PATH.relative_to(ROOT)}.", "success")
        return redirect(url_for("profile"))

    try:
        current = json.loads(PROFILE_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        current = {}

    return render_template(
        "profile.html",
        profile=current,
        profile_path=str(PROFILE_PATH.relative_to(ROOT)),
        username=session.get("username"),
    )


def main() -> None:
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    # threaded=True is the default on dev server; ensure SSE works alongside other requests.
    app.run(host="127.0.0.1", port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    main()
