"""
Deployment-shape guards — not logic tests. The controller Dockerfile
COPYs each module explicitly, so a new em_*.py that works fine on bare
metal crash-loops the container at import time if the COPY line is
forgotten (bitten by em_scenes.py 2026-07-10 and em_oww_models.py
2026-07-19).
"""

import re
from pathlib import Path

CONTROLLER = Path(__file__).resolve().parents[1]
ROOT = CONTROLLER.parent


def test_dockerfile_copies_every_controller_module():
    dockerfile = (CONTROLLER / "Dockerfile").read_text()
    copied = set(re.findall(r"^COPY\s+(\S+\.py)\s", dockerfile, re.M))
    modules = {p.name for p in CONTROLLER.glob("em_*.py")} | {"version.py"}
    missing = sorted(modules - copied)
    assert not missing, (
        f"Dockerfile is missing COPY lines for {missing} — the container "
        f"will crash-loop at import time"
    )


def test_dashboard_bundle_is_cache_busted():
    """
    /dashboard must not hand the browser a bare /static/dashboard.js URL.

    aiohttp's add_static sends Last-Modified and ETag but no Cache-Control, so
    browsers apply heuristic freshness and serve a stale bundle without
    revalidating. That failure is invisible server-side — deploy correct, file
    correct, compiled bundle correct, browser showing the previous UI — so it
    reads as "my change didn't work" and sends you hunting in the wrong place.
    Asserted at the source level because the alternative is starting an aiohttp
    app, which this suite deliberately does not do.
    """
    import re
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent / "em_api.py").read_text()
    handler = src[src.index("async def _serve_dashboard"):]
    handler = handler[:handler.index("\nasync def ", 1)]

    assert "dashboard.js?v=" in handler, \
        "the bundle URL must carry a cache-busting token"
    assert "no-cache" in handler, \
        "dashboard.html itself must be revalidated, or the new URL is never seen"
    # A version-string token would not change between two local "dev" builds;
    # mtime changes on every rebuild.
    assert "st_mtime" in handler, \
        "cache-bust on the bundle's mtime, not on a version string"


def test_turn_dashboard_uses_streaming_stage_boundaries():
    jsx = (CONTROLLER / "static" / "dashboard.jsx").read_text()
    start = jsx.index("const TURN_STAGES")
    end = jsx.index("function TurnObservability", start)
    block = jsx[start:end]

    for label in ("STT", "HA response", "TTS startup", "TTS playback"):
        assert f"label: '{label}'" in block
    assert "duration(ttsUrl, sttMark)" in block
    assert "duration(firstPcm, ttsUrl)" in block
    assert "duration(playbackDrained, firstPcm)" in block
    assert "vad_end_ms" not in block
    assert "total_ms || 0" not in block, \
        "missing timing data must stay unavailable, not become the whole turn"


def test_turn_dashboard_renders_missing_timings_as_unavailable():
    jsx = (CONTROLLER / "static" / "dashboard.jsx").read_text()
    assert "const fmtS = ms => ms == null ? '—'" in jsx
    assert "const playbackDrained = mark(t.playback_drained_ms)" in jsx
    assert "const ttsPlayback = duration(playbackDrained, firstPcm)" in jsx
    assert "duration(total, firstPcm)" not in jsx
    assert "const scale = Math.max(1, ...recent.map(t => turnSegments(t).shown))" in jsx
    assert "seg[s.key] / scale * 100" in jsx
    assert "Math.max(3000" not in jsx


def test_home_dashboard_has_authenticated_fleet_query_history():
    api = (CONTROLLER / "em_api.py").read_text()
    jsx = (CONTROLLER / "static" / "dashboard.jsx").read_text()
    assert 'app.router.add_get("/api/activity/turns"' in api
    handler = api[api.index("async def _get_fleet_turns"):]
    handler = handler[:handler.index("\nasync def ", 1)]
    before = api[:api.index("async def _get_fleet_turns")]
    assert before.rstrip().endswith("@auth.require_auth")
    assert "function FleetActivity" in jsx
    assert "Query history" in jsx
    assert "API.get(queryPath" in jsx


def test_fleet_query_history_has_filters_and_cursor_pagination():
    jsx = (CONTROLLER / "static" / "dashboard.jsx").read_text()
    start = jsx.index("function FleetActivity")
    end = jsx.index("// ─── Connectivity tab", start)
    block = jsx[start:end]
    for text in (
        "All devices", "All outcomes", "Last 24 hours", "Last 7 days",
        "Last 30 days", "All retained", "25 at a time", "50 at a time",
        "100 at a time", "Load {pageSize} older", "next_cursor",
    ):
        assert text in block
    assert "setInterval" in block and "30000" in block
    assert "historyMaxHeight" not in block
    assert "overflowY: 'auto'" not in block


def test_dashboard_uses_local_roboto_for_proportional_text():
    dockerfile = (CONTROLLER / "Dockerfile").read_text()
    dashboard = (CONTROLLER / "static" / "dashboard.html").read_text()
    landing = (CONTROLLER / "static" / "index.html").read_text()
    jsx = (CONTROLLER / "static" / "dashboard.jsx").read_text()
    assert "roboto.woff2" in dockerfile
    assert "font-family: 'Roboto'" in dashboard
    assert "font-family: 'Roboto'" in landing
    assert "DM Mono" not in dashboard + landing + jsx
    assert "dm-mono" not in dockerfile
    assert "DM Sans" not in dashboard + landing + jsx


def test_fleet_recordings_are_scoped_to_each_turns_device():
    jsx = (CONTROLLER / "static" / "dashboard.jsx").read_text()
    start = jsx.index("function TurnObservability")
    end = jsx.index("function FleetActivity", start)
    block = jsx[start:end]
    assert "const turnDeviceId = t => t.device_id || deviceId" in block
    assert "`${turnDeviceId(t)}:${t.turn_id}`" in block
    assert "`/api/devices/${turnDeviceId(t)}/turns/${t.turn_id}/audio`" in block


def test_debug_audio_has_separate_authenticated_playback_controls():
    api = (CONTROLLER / "em_api.py").read_text()
    jsx = (CONTROLLER / "static" / "dashboard.jsx").read_text()
    assert 'audio/{kind}' in api
    assert 'kind not in ("stt", "loopback")' in api
    assert "audio/${kind}`" in jsx
    assert "toggleDebugAudio(t, 'stt')" in jsx
    assert "toggleDebugAudio(t, 'loopback')" in jsx
    assert "downloadDebugAudio(t, 'stt')" in jsx
    assert "downloadDebugAudio(t, 'loopback')" in jsx


def test_turn_details_expand_one_row_at_a_time():
    jsx = (CONTROLLER / "static" / "dashboard.jsx").read_text()
    start = jsx.index("function TurnObservability")
    end = jsx.index("function FleetActivity", start)
    block = jsx[start:end]
    assert "const [expanded, setExpanded] = useState(null)" in block
    assert "current === i ? null : i" in block
    assert "expanded === i &&" in block
    assert "Hover detail" not in block


def test_query_history_displays_global_query_id():
    jsx = (CONTROLLER / "static" / "dashboard.jsx").read_text()
    start = jsx.index("function TurnObservability")
    end = jsx.index("function FleetActivity", start)
    block = jsx[start:end]
    assert "t.query_id != null" in block
    assert "`#${t.query_id}`" in block


def test_debug_audio_capture_uses_ingress_fanout_and_exact_boundaries():
    controller = (CONTROLLER / "em_controller.py").read_text()
    esphome = (CONTROLLER / "em_esphome.py").read_text()
    data = controller[controller.index("async def handle_data"):]
    data = data[:data.index("# ─── Router")]
    assert "device.capture_query_mic(payload)" in data
    payload_branch = data[data.index("payload = raw[MIC_HEADER_LEN:]"):]
    assert payload_branch.index("device.capture_query_mic(payload)") < \
        payload_branch.index("q = device.voice_queue")
    stt = esphome[esphome.index("VOICE_ASSISTANT_STT_END"):]
    stt = stt[:stt.index("VOICE_ASSISTANT_INTENT_END")]
    assert "mark_query_stt_final()" in stt
    assert "mark_query_uplink_drained()" in esphome
    assert "finish_query_stt_capture()" in controller
    assert "query_capture_tts_enabled" in stt
    assert "mic_restart_continuous()" in stt
    persist = esphome[esphome.index("async def _persist_turn"):]
    assert "await _finish_debug_audio(device, turn_record)" in persist
    finish = esphome[esphome.index("async def _finish_debug_audio"):
                   esphome.index("async def _persist_turn")]
    assert 'if kind == "loopback"' not in finish


def test_loopback_capture_keeps_mic_running_without_enabling_barge_in():
    controller = (CONTROLLER / "em_controller.py").read_text()
    start = controller.index("async def post_turn_play_esphome")
    end = controller.index("# P0-1:", start)
    block = controller[start:end]
    assert "elif not device.query_capture_tts_enabled:" in block
    assert "await device.mic_stop()" in block


def test_device_mic_wire_format_is_16khz_mono_s16le():
    data = (ROOT / "device" / "internal" / "client" / "data.go").read_text()
    beam = (ROOT / "device" / "internal" / "beamformer" / "beamformer.go").read_text()
    assert "vadOwwChunkBytes = 1280 * 2 // 2560 bytes = 80ms" in data
    assert "frame := make([]byte, 3+len(payload))" in data
    assert "frame[0] = frameTypeMic" in data
    assert "copy(frame[3:], payload)" in data
    assert "out := make([]byte, n*2)" in beam
    assert "out[i*2] = byte(uint16(v))" in beam
    assert "out[i*2+1] = byte(uint16(v) >> 8)" in beam


def test_release_notes_survive_the_whole_relay():
    """
    Release notes have to make it through four places to be useful: captured
    from the GitHub response, persisted, re-read into the cache after a
    restart, and rendered. Miss any one and the dashboard shows a version
    number with no way to judge it — which is the state this replaced.

    The restart path is the one worth pinning: the in-memory cache is
    populated from the DB when cold, so notes omitted there would appear on
    first poll and silently vanish on every controller restart until the next
    one.
    """
    from pathlib import Path
    api = (Path(__file__).resolve().parent.parent / "em_api.py").read_text()

    fetch = api[api.index("async def _fetch_latest_release"):]
    fetch = fetch[:fetch.index("\nasync def ", 1)]
    assert 'release.get("body")' in fetch or '.get("body")' in fetch, \
        "the GitHub release body must be captured"
    assert 'set_config("latest_notes"' in fetch, "notes must be persisted"

    cached = api[api.index("    # Load from DB cache"):]
    cached = cached[:cached.index("\nasync def ", 1)] if "\nasync def " in cached else cached
    assert 'get_config("latest_notes"' in cached, \
        "the DB-cache path must restore notes, or they vanish on restart"

    jsx = (Path(__file__).resolve().parent.parent / "static" / "dashboard.jsx").read_text()
    assert "release.notes" in jsx, "the dashboard must render the notes"


def test_release_workflow_publishes_the_tag_annotation():
    """
    The notes shown in the dashboard come from the annotated tag, so the
    workflow must publish that rather than only GitHub's generated commit
    list. If this drifts, every future release silently shows a commit dump to
    whoever is deciding whether to update.
    """
    from pathlib import Path
    wf = (Path(__file__).resolve().parent.parent.parent
          / ".github" / "workflows" / "release.yml").read_text()
    assert "body_path:" in wf, "the release must publish notes from a file"
    assert "%(contents)" in wf, "notes must come from the tag annotation"


def test_every_device_payload_has_an_update_path():
    """
    A payload installed only at provisioning drifts forever.

    This has now bitten twice: start_server.sh (Lounge was a revision behind
    Office, 2026-07-11) and the debloat pair (round 2 added a package and every
    fielded device needed a manual push, 2026-07-30). Both fixes were the same
    shape — an md5-compared sync riding the OTA — so this asserts that every
    file in device_payloads/ is named by a sync function, and fails when a
    fourth payload is added without one.
    """
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    api = (root / "em_api.py").read_text()
    payloads = sorted(p.name for p in (root / "device_payloads").iterdir() if p.is_file())
    assert payloads, "device_payloads/ is empty — has it moved?"

    for name in payloads:
        assert name in api, (
            f"{name} has no update path: nothing in em_api.py references it. "
            f"A payload installed only by the provisioning wizard drifts on every "
            f"device already in the field."
        )


def test_debloat_sync_reconciles_both_halves():
    """
    The debloat is a boot script AND a pm-hide list. Round 2 added a *package*,
    so a sync that only refreshed the script would have looked like it worked
    and changed nothing on any device.
    """
    from pathlib import Path
    api = (Path(__file__).resolve().parent.parent / "em_api.py").read_text()
    fn = api[api.index("async def _sync_debloat"):]
    fn = fn[:fn.index("\nasync def ", 1)] if "\nasync def " in fn[1:] else fn

    assert "echomuse-debloat.sh" in fn, "the boot script half must be synced"
    assert "_debloat_packages()" in fn, "the pm-hide half must be reconciled"
    assert "pm hide" in fn, "drifted packages must actually be hidden"
    # Rename-based replacement: the running shell keeps the old inode.
    assert "mv " in fn and ".new" in fn, \
        "the script must be replaced by rename, not written in place"
    # md5 both before (skip when in sync) and after (verify the transfer).
    assert fn.count("md5") >= 2, "sync must md5-compare and md5-verify"


def test_debloat_reachable_without_an_ota():
    """
    The OTA-time sync cannot reach a device already on the latest firmware —
    which is the exact case that exposed the gap. A manual trigger is required,
    not a nicety.
    """
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    api = (root / "em_api.py").read_text()
    assert '"/api/devices/{id}/debloat"' in api, "no manual debloat endpoint registered"
    jsx = (root / "static" / "dashboard.jsx").read_text()
    assert "/debloat`" in jsx, "the dashboard must be able to trigger it"


def test_stale_release_cache_is_not_returned_when_it_has_aged_out():
    """
    _get_cached_release used to fire a refresh into the background and return
    the STALE value. Two consequences, both seen on 2026-07-30: the dashboard
    reported "there's an update" only after someone pressed Check now, and an
    OTA pushed v2.9.9 while v2.9.10 was the current release.

    The refresh is now awaited when the DB cache has aged past the check
    interval, falling back to the stale value only if the fetch fails.
    """
    from pathlib import Path
    api = (Path(__file__).resolve().parent.parent / "em_api.py").read_text()
    fn = api[api.index("async def _get_cached_release"):]
    fn = fn[:fn.index("\nasync def ", 1)]

    assert "await _fetch_latest_release()" in fn, \
        "an aged-out cache must be refreshed synchronously, not in the background"
    assert "asyncio.create_task(_fetch_latest_release())" not in fn, \
        "fire-and-forget refresh returns the stale value to this caller"


def test_release_change_is_pushed_to_open_dashboards():
    """A tab already showing the Updates panel should not sit on the old
    version until someone reloads."""
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    api = (root / "em_api.py").read_text()
    assert '"type":         "release_update"' in api or '"release_update"' in api, \
        "a release change must be broadcast on the event stream"
    jsx = (root / "static" / "dashboard.jsx").read_text()
    # And the tab must keep asking, as a fallback for a missed event.
    upd = jsx[jsx.index("if (tab !== 'updates') return;"):]
    assert "setInterval" in upd[:600], \
        "the Updates tab must refresh while open, not fetch once on entry"


def test_streamed_playback_waits_for_the_device_not_a_computed_sleep():
    """
    Turn playback must end when the DEVICE says its buffer drained, never on an
    `audio_duration - elapsed` estimate.

    That estimate was removed on 2026-07-24: it has no visibility of the
    device's own buffer and cleared the ring 6.1s early on Retreat, 3.2s on
    Lounge. The streaming path reintroduced it (PR #47) where it is worse still
    — streaming already consumes most of the audio duration, so the remainder
    computes to ~0 and the wait disappears while up to ~5.5s is queued in
    audioChanDepth.
    """
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "em_controller.py").read_text()
    fn = src[src.index("async def _run_streaming_post_turn_playback"):]
    fn = fn[:fn.index("\nasync def ", 1)]

    assert "playback_done" in fn, \
        "streamed playback must await the device's playback_stats"
    assert "asyncio.sleep(remaining)" not in fn, \
        "the computed drain estimate was removed on 2026-07-24 — do not restore it"


def test_send_ms_stays_socket_write_time():
    """
    send_ms is documented as socket-write time that completes near-instantly
    however slow the link is — "never read it as delivery; that mistake cost a
    whole investigation on 2026-07-20". Timing the whole streaming loop instead
    folds HA's synthesis time in and makes it read exactly like delivery.
    """
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "em_controller.py").read_text()
    fn = src[src.index("async def stream_speaker_chunks"):]
    fn = fn[:fn.index("\n    async def ", 1) if "\n    async def " in fn[1:] else len(fn)]
    assert "send_seconds" in fn, \
        "socket-write time must be accumulated around the send calls"

    caller = src[src.index("async def _run_streaming_post_turn_playback"):]
    caller = caller[:caller.index("\nasync def ", 1)]
    assert "device.playback_send_ms = send_ms" in caller, \
        "send_ms must come from accumulated write time, not the loop duration"


def test_meter_ring_is_raised_when_audio_starts_not_at_playback_setup():
    """
    The meter pattern renders the live speaker RMS, so it draws an UNLIT ring
    until the device's ALSA write actually begins.

    On the buffered path that was invisible: the TTS fetch had already
    completed, so frames flushed at socket speed and audio began almost at
    once. Streaming moves fetch+decode inside playback, so raising the meter at
    setup time leaves the ring dark from the end of the spinner until HA
    returns audio — seconds on a slow response, and indistinguishable from a
    failed turn (user report 2026-07-31).

    The spinner must therefore stay up until the meter has something to show.
    """
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "em_controller.py").read_text()
    fn = src[src.index("async def post_turn_play_esphome"):]
    fn = fn[:fn.index("\n            # P0-1")]

    assert "_meter_at_playback_start(pcm_chunks" in fn, \
        "the meter must be gated on the first audio reaching the device"

    # A meter send is legitimate inside the nested helpers (_meter_on, and the
    # dead-man refresher). What must not exist is one in the function's OWN
    # body — that runs at setup, before any audio. Nested bodies are indented
    # deeper, so indentation is the discriminator.
    assert "\n" + " " * 20 + "await device.send_led_anim(meter)" not in fn, \
        ("the meter is being raised at playback setup — on the streaming path "
         "that is before HA has returned any audio, leaving the ring dark")


def test_meter_gate_fires_for_responses_shorter_than_the_prime_window():
    """
    A response shorter than SPEAKER_PRIME_SECONDS never reaches the byte
    threshold — the device starts playing it at EOS instead. Exhaustion must
    fire the callback too, or short answers play with no ring at all.
    """
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "em_controller.py").read_text()
    fn = src[src.index("async def _meter_at_playback_start"):]
    fn = fn[:fn.index("\nasync def ", 1)]

    body = fn.split("async for")[1]
    assert "if not fired:" in body.rsplit("\n", 4)[-4:][0] or "if not fired" in body, \
        "the generator must fire on exhaustion for sub-prime-length responses"
    assert "SPEAKER_PRIME_SECONDS" in fn, \
        "the threshold must track the device's actual prime window"


def test_playback_started_replaces_setup_timestamp_and_keeps_fallback():
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "em_esphome.py").read_text()
    assert "def playback_started(self, age_ms" in src
    assert "trace.playback_started(age_ms)" in src
    assert "satellite = server.get_satellite()" in src
    assert "trace = satellite._trace" in src
    assert "if trace and trace.t_playback_ms < 0" in src
    assert 'elif msg_type == "playback_started"' in (
        (Path(__file__).resolve().parent.parent / "em_controller.py").read_text()
    )


def test_esphome_requests_provider_native_tts_format():
    jsx = (CONTROLLER / "em_esphome.py").read_text()
    start = jsx.index("_fmt = dict(")
    block = jsx[start:jsx.index("yield api_pb2.ListEntitiesMediaPlayerResponse", start)]
    assert 'format="wav"' in block
    assert "sample_rate=24000" in block


def test_controller_update_is_advisory_only():
    """
    The dashboard may TELL you a newer controller exists; it must never offer
    to apply it.

    The controller is a container the user owns and updates with their own
    docker tooling. An in-app update would have to restart the process serving
    the page, mid-request, with no way to report the outcome — and it is an
    explicit product decision (Wil, 2026-07-31) that this stays out of the
    interface. The notice is information for a decision the user takes
    elsewhere.
    """
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent

    jsx = (root / "static" / "dashboard.jsx").read_text()
    start = jsx.index("{/* Controller update notice.")
    banner = jsx[start:jsx.index("{/* Summary */}", start)]
    for forbidden in ("API.post", "API.put", "API.delete", "onClick={doUpdate"):
        assert forbidden not in banner, (
            f"the controller update notice must not perform actions, found "
            f"{forbidden!r} — updating is the user's docker command to run"
        )

    api = (root / "em_api.py").read_text()
    assert 'add_get("/api/releases/controller"' in api, \
        "the controller release endpoint must be read-only (GET)"
    for verb in ("add_post", "add_put", "add_delete"):
        assert f'{verb}("/api/releases/controller' not in api, \
            f"{verb} on /api/releases/controller would make the update actionable"


def test_controller_notes_come_from_the_tag_annotation():
    """
    controller-v* tags ship a GHCR image and no GitHub Release (CLAUDE.md,
    "Versioning / releases"), so the notes must be read from the annotated
    tag object. Reading them from the releases list would return the newest
    DEVICE firmware release instead — right shape, wrong product.
    """
    from pathlib import Path
    api = (Path(__file__).resolve().parent.parent / "em_api.py").read_text()
    fn = api[api.index("async def _fetch_controller_release"):]
    fn = fn[:fn.index("\nasync def ", 1)]
    assert "GITHUB_TAGS_URL" in fn and "GITHUB_TAG_OBJECT_URL" in fn, \
        "controller notes must come from the tag annotation, not /releases"
    assert "GITHUB_API_URL" not in fn, \
        "that is the device firmware release feed, not the controller's"


def test_every_db_call_in_em_api_exists():
    """
    A typo'd db.<name> is invisible to pyflakes (it is a valid attribute
    expression) and raises AttributeError only on the request that uses it.
    That is the same shape as the NameError which stopped wake word
    fleet-wide on 2026-07-30: green CI, clean logs, broken at runtime.

    Written after db.get_global_config() — a function that has never existed —
    reached a live endpoint.
    """
    import ast
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent

    tree = ast.parse((root / "em_api.py").read_text())
    used = {
        n.attr for n in ast.walk(tree)
        if isinstance(n, ast.Attribute)
        and isinstance(n.value, ast.Name) and n.value.id == "db"
    }
    db_tree = ast.parse((root / "em_db.py").read_text())
    defined = {
        n.name for n in db_tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    } | {
        t.id for n in db_tree.body if isinstance(n, ast.Assign)
        for t in n.targets if isinstance(t, ast.Name)
    } | {
        # Annotated module constants, e.g. `MIGRATIONS: list[str] = [...]`.
        # Missing these produced a false positive on db.MIGRATIONS, which is
        # the failure mode that gets a guard disabled rather than fixed.
        n.target.id for n in db_tree.body
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
    }
    missing = sorted(used - defined)
    assert not missing, f"em_api.py calls db.{{{', '.join(missing)}}} which em_db.py does not define"


def test_the_wake_word_asset_wizard_step_is_mandatory():
    """
    Wil's call, overriding my "make it skippable": every provisioned device
    carries the runtime.

    The assets are not in the firmware, so a device without them advertises
    the oww_shadow capability while being unable to use it — the exact "I
    enabled it and nothing happened" this feature exists to remove. It also
    auto-runs: a button someone can leave unpressed is not mandatory.
    """
    from pathlib import Path
    jsx = (Path(__file__).resolve().parent.parent / "static" / "dashboard.jsx").read_text()

    steps = jsx[jsx.index("const _WIZARD_STEPS = ["):]
    steps = steps[:steps.index("\n];")]
    assert "'install_oww'" in steps, "the wake word asset step is missing from the wizard"

    idx = steps.count("{ id:", 0, steps.index("'install_oww'")) - 1
    auto = jsx[jsx.index("const autoSteps = new Set(["):]
    auto = auto[:auto.index(")")]
    assert str(idx) in auto, (
        f"step {idx} (install_oww) must auto-run — a step that needs a click "
        f"is one a user can skip"
    )

    runner = jsx[jsx.index("async function runInstallOwwAssets"):]
    runner = runner[:runner.index("\n  async function ", 1)]
    assert "a.md5" in runner and "throw new Error" in runner, \
        "the push must verify md5 and fail loudly — a truncated file fails later at dlopen"


def test_turn_end_reports_real_media_state_not_a_hardcoded_idle():
    """
    Issue #53: "the esphome media player reports that it is idle even though
    the music continues to play on the echo."

    Every voice turn ended by asserting MediaPlayerState.IDLE regardless of
    what the media player was doing. The feed announces PLAYING exactly once,
    when the decoder starts, so this IDLE arrived afterwards and became HA's
    last word — the entity showed a play arrow over audible music, and nothing
    ever corrected it.

    _media_state_msg() exists for precisely this ("current media_player state
    as HA should see it — em_player truth") and was being bypassed.
    """
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "em_esphome.py").read_text()

    fn = src[src.index("        finally:\n            # Signal HA that the satellite has finished"):]
    fn = fn[:fn.index("self._turn_active    = False")]

    assert "self._media_state_msg()" in fn, \
        "the turn must report the real media state at the end"
    assert "state=MediaPlayerState.IDLE" not in fn, (
        "a hardcoded IDLE at turn end overwrites the feed's PLAYING and "
        "leaves HA showing idle over audible music"
    )


def test_no_unjustified_hardcoded_media_state():
    """
    Forbid the SHAPE, not just the instances.

    A hardcoded MediaPlayerState sent to HA asserts what the player is doing
    without asking em_player, so it is wrong whenever the guess is wrong — and
    it wins, because the feed announces PLAYING exactly once, at the start.

    This has now been the same bug twice: the turn-end IDLE (#53, "reports
    idle even though the music continues to play"), and then IDLE on every
    device volume report, which told Music Assistant the music had stopped
    while it was audibly playing. Fixing instances one at a time is how the
    second one survived the first fix, so this pins the rule.

    Two remain legitimate and are named explicitly:
      - PLAYING for play_media, documented as optimistic — the feed pushes
        the authoritative state moments later.
      - ANNOUNCING, a genuine transition with no em_player equivalent.

    Anything else must go through _media_state_msg(), which reads em_player
    truth.
    """
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "em_esphome.py").read_text()

    allowed = {"MediaPlayerState.PLAYING", "MediaPlayerState.ANNOUNCING"}
    found = [
        line.strip()
        for line in src.splitlines()
        if "state=MediaPlayerState." in line
    ]
    offenders = [
        ln for ln in found
        if not any(a in ln for a in allowed)
    ]
    assert not offenders, (
        f"hardcoded media state(s) {offenders} — use _media_state_msg() so the "
        f"entity reflects what em_player is actually doing"
    )


def test_every_deliberate_cancel_also_flushes_the_speaker():
    """
    Cancelling a turn must stop the AUDIO, not just our end of it.

    cancel_event aborts the controller's feed. It cannot touch what is already
    on the device — up to ~5.5s sits in audioChanDepth — so without a
    speaker_flush the ring clears and the device carries on talking after you
    have visibly cancelled it. Reported 2026-08-01 for the action button,
    which was the one deliberate cancel missing it while mute and barge-in
    both had it.

    Forbidding the shape rather than fixing the instance: this is the second
    bug of exactly this kind today (the other was a hardcoded MediaPlayerState
    in one of three places), and fixing them one at a time is how the second
    one survived the first.
    """
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "em_controller.py").read_text()

    # Each deliberate cancel site, identified by its log line / guard, paired
    # with how far to look for the flush that must accompany it.
    sites = {
        "Dot button — cancelling voice turn": 900,
        "Muted during active turn": 900,
    }
    for marker, window in sites.items():
        i = src.find(marker)
        assert i != -1, f"cancel site {marker!r} not found — has it been renamed?"
        block = src[i:i + window]
        assert "cancel_event.set()" in block, f"{marker}: no cancel"
        assert "speaker_flush" in block, (
            f"{marker}: cancels the turn but never flushes the device speaker — "
            f"the response will keep playing after the turn is cancelled"
        )


def test_supervisor_log_path_matches_between_script_and_controller():
    """
    The supervisor writes its decisions to a persistent path and the
    controller reads them back from it. Two languages, one path — if they
    drift, the fetch silently returns nothing and a failed update stays
    unexplained, which is the exact failure this feature exists to remove.
    """
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent

    script = (root / "device_payloads" / "start_server.sh").read_text()
    api = (root / "em_api.py").read_text()

    import re
    m = re.search(r"^SUP_LOG=(\S+)", script, re.M)
    assert m, "start_server.sh no longer defines SUP_LOG"
    script_path = m.group(1)

    m = re.search(r'^SUPERVISOR_LOG = "([^"]+)"', api, re.M)
    assert m, "em_api.py no longer defines SUPERVISOR_LOG"
    assert m.group(1) == script_path, (
        f"supervisor log path drifted: script writes {script_path}, "
        f"controller reads {m.group(1)}"
    )


def test_supervisor_log_is_persistent_and_bounded():
    """
    Two properties it cannot lose.

    PERSISTENT: under /data. /tmp is RAM-backed, so a log there is wiped by
    the reboot used to recover from the very failure it would explain.

    BOUNDED: these devices have ~350MB free and no operator. A crash-loop
    writing every few seconds must never be able to fill /data — so the trim
    happens BEFORE the append, not after.
    """
    from pathlib import Path
    import re
    script = (Path(__file__).resolve().parent.parent
              / "device_payloads" / "start_server.sh").read_text()

    m = re.search(r"^SUP_LOG=(\S+)", script, re.M)
    assert m.group(1).startswith("/data/"), \
        "the supervisor log must live on persistent storage, not /tmp"

    fn = script[script.index("sup_log() {"):]
    fn = fn[:fn.index("\n}")]
    trim = fn.index("SUP_MAX")
    append = fn.index('>> "$SUP_LOG"')
    assert trim < append, \
        "the size check must run before the append, or a crash-loop outruns it"


def test_a_failed_update_asks_for_the_supervisor_log():
    """
    The fetch cannot happen at failure time — the device is gone, which IS the
    problem. So every failure path must record that an explanation is owed,
    and the connect path must collect it.
    """
    from pathlib import Path
    api = (Path(__file__).resolve().parent.parent / "em_api.py").read_text()

    # The failure paths live in _run_update, which is what awaits
    # _monitor_reconnect and decides what its result means.
    monitor = api[api.index("async def _run_update"):]
    monitor = monitor[:monitor.index("\nasync def ", 1)]
    assert monitor.count("_supervisor_log_wanted.add") >= 2, (
        "both update-failure paths (auto-rollback and timeout) must request "
        "the supervisor log"
    )

    connect = api[api.index("async def notify_device_connected"):]
    connect = connect[:connect.index("\nasync def ", 1)]
    assert "_collect_supervisor_log" in connect, \
        "nothing collects the supervisor log when the device comes back"


def test_data_reconnect_grace_is_per_stream_not_per_frame():
    """
    A dropped data connection mid-stream should cost a pause, not the rest of
    the audio (#28, @kopiro — long read-aloud responses truncated by a brief
    Wi-Fi blip).

    The budget must be spent DOWN across the stream, never a fresh wait on
    each call. send_data runs once per audio period, so a per-frame wait means
    a device that is genuinely gone stalls every remaining frame in turn — a
    stream that should abort in seconds instead drains for hours holding the
    voice lock. That failure is worse than the truncation it replaces, which
    is why the shape is pinned rather than the constant.
    """
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "em_controller.py").read_text()

    fn = src[src.index("    async def send_data(self, data: bytes):"):]
    fn = fn[:fn.index("\n    async def ", 1)]

    assert "_data_grace_left -=" in fn, (
        "the reconnect grace must be spent down across the stream; a wait that "
        "does not decrement is a per-frame stall"
    )
    assert "_data_grace_left > 0" in fn, \
        "send_data must stop waiting once the stream's budget is exhausted"

    # Every path that streams audio has to arm it, or the budget is stale from
    # whatever ran last.
    for stream_fn in ("async def stream_speaker(self",
                      "async def stream_speaker_chunks(self"):
        body = src[src.index(stream_fn):]
        body = body[:body.index("\n    async def ", 1)]
        assert "begin_data_stream()" in body, \
            f"{stream_fn} does not arm the reconnect grace"


def test_support_bundle_attributes_metrics_to_a_device():
    """
    `db.get_device_metrics` builds its own result dicts and does NOT include
    the device, so the support handler must attach it. Without that, every
    device's hourly CPU, memory and RTT pool into one flat anonymous list —
    present in the bundle, useless for diagnosis, and wrong in the quiet way
    where the file still looks full of data. Shipped like that in #63.
    """
    src = (CONTROLLER / "em_api.py").read_text()
    body = src.split("_get_support_bundle")[-1]
    call = re.search(r"metrics \+= \[(.*?)\]\n", body, re.S)
    assert call, "could not find where the support bundle collects metrics"
    assert "device_id" in call.group(1), (
        "support bundle metrics rows must carry device_id — "
        "get_device_metrics does not return it"
    )


def test_support_bundle_redacts_account_names():
    """
    Account names reach the bundle through ordinary log prose ("Shell session
    opened by wil"), which no quote, URL or identifier rule matches. The
    handler must therefore pass the user table to em_support; the redaction
    itself is tested in test_support.py.
    """
    src = (CONTROLLER / "em_api.py").read_text()
    # "return web.Response", not "web.Response" — the latter is the handler's
    # own return annotation and slices the body away to nothing.
    body = src.split("_get_support_bundle")[-1].split("return web.Response")[0]
    assert "accounts=" in body and "get_all_users" in body, (
        "the support bundle must pass the real accounts to em_support — "
        "there is nothing in a log line to pattern-match them by"
    )
    assert '"role"' in body, (
        "accounts must carry the role: a name is replaced by <admin>, and a "
        "positional alias would be one-to-one with a real person"
    )


def test_the_music_feed_reads_its_lead_per_chunk():
    """
    A voice turn lowers the music feed's lead so the response gets the shared
    data plane (TURN_LEAD_S). That only works if the pacing loop reads the
    CURRENT lead each time round — capturing LEAD_S once, or referring to the
    module constant, silently restores the old behaviour and the fix becomes
    a no-op with every test still passing.
    """
    src = (CONTROLLER / "em_player.py").read_text()
    feed = src.split("async def _feed")[-1]
    pacing = re.search(r"ahead = sent / BYTES_PER_SEC.*?await asyncio\.sleep\(([^)]*)\)",
                       feed, re.S)
    assert pacing, "could not find the feed's pacing sleep"
    window = feed[:pacing.end()]
    assert "self.lead_s" in window, (
        "the pacing loop must read self.lead_s, not the LEAD_S constant — "
        "otherwise lowering the lead for a voice turn does nothing"
    )


def _fn_body(src: str, name: str) -> str:
    """Slice one async def out of a module's source, up to the next top-level def."""
    start = src.index(f"async def {name}")
    rest  = src[start + 1:]
    end   = rest.index("\nasync def ") if "\nasync def " in rest else len(rest)
    return src[start:start + 1 + end]


def test_firmware_transfer_is_verified_by_md5_not_by_an_exit_status():
    """
    TRANSFER_OK only ever proved that the decode pipeline and chmod exited 0 —
    not that the bytes on the device match the bytes we sent (#76).

    That matters because a corrupt binary and a genuinely broken one produce
    the SAME observable: three fast exits, a symlink flip, and a device back on
    its old version. Shipping an unverified binary therefore costs a reboot and
    a rollback to learn nothing at all.
    """
    src = (CONTROLLER / "em_api.py").read_text()
    fn  = _fn_body(src, "_stream_file_to_device")

    assert "hashlib.md5(data).hexdigest()" in fn, (
        "the transfer must hash what it actually sent — hashing anything else "
        "verifies the wrong thing"
    )
    assert ".part" in fn and "mv " in fn, (
        "bytes must land in .part and be renamed only once verified, so a bad "
        "transfer leaves the destination as it was"
    )
    # The rename must be conditional on the comparison, not merely nearby.
    assert re.search(r"case .*GOT.*in .*want.*mv ", fn, re.S), (
        "the rename must be guarded by the md5 comparison"
    )
    assert "rm -f" in fn, "a failed verification must remove the .part"

    # Anchored on the CALL, not the function body: the docstring explains why
    # require_verify is set, so a body-wide search passes on the prose alone
    # while the argument is gone (caught by reintroducing exactly that).
    slot = _fn_body(src, "_stream_binary_to_slot")
    call = slot[slot.index("return await _stream_file_to_device"):]
    assert "require_verify=True" in call, (
        "firmware is the payload where an unverifiable transfer must fail "
        "rather than proceed — it is the one we are about to boot"
    )


def test_a_corrupt_binary_never_reaches_the_symlink_flip():
    """
    The real win of verifying is not the error message, it is the ordering:
    a mismatch must be caught while the device is still running fine, not
    after it has taken a reboot and a rollback to tell us the same thing.
    """
    src = (CONTROLLER / "em_api.py").read_text()
    ota = src[src.index("_stream_binary_to_slot(live, binary"):]
    ota = ota[:ota.index("_monitor_reconnect")]

    guard = ota.index("if not ok:")
    flip  = ota.index("ln -sf")
    assert guard < flip, (
        "the transfer result must be checked BEFORE the symlink flip"
    )
    assert "return" in ota[guard:flip], (
        "a failed transfer must return, not fall through to the flip — "
        "otherwise verification changes the log message and nothing else"
    )


def test_ota_checks_free_space_before_writing_anything():
    """
    The OTA path had no space check at all, unlike the asset path.

    Two traps, both already paid for elsewhere: read the figure with
    parse_free_mb rather than an awk field index (busybox wraps a long
    filesystem name onto its own line, so $4 is the PERCENTAGE on these
    devices), and treat an unreadable df as "carry on" rather than as a full
    disk — refusing on an unparsed reading blocks updates on any device whose
    df we have not seen.
    """
    src = (CONTROLLER / "em_api.py").read_text()
    ota = src[src.index("inactive_slot = "):src.index("_stream_binary_to_slot(live, binary")]

    assert "parse_free_mb" in ota, (
        "free space must be read with parse_free_mb, never an awk field index"
    )
    # Comments are stripped first: the trap is worth explaining in a comment,
    # and a test that reads its own warning as the bug is a test that can only
    # be silenced by deleting the explanation.
    code = "\n".join(l for l in ota.splitlines() if not l.lstrip().startswith("#"))
    assert "awk" not in code, "an awk field index reads the percentage on these devices"
    assert "free_mb is not None and free_mb <" in ota, (
        "an unknown reading must not be compared as if it were a number"
    )


def test_a_transfer_never_deletes_the_destination_before_sending():
    """
    For firmware the destination IS the rollback slot, so deleting it up front
    means a transfer that fails early leaves the device with a good active
    slot and an empty partner — and a later crash-loop flips the symlink onto
    nothing. Three Dots hit exactly that in #121, while being told the slot
    had been left untouched.

    The `.part` discipline only protects `dest` from a CORRUPT transfer. It
    cannot protect it from being removed before the transfer starts.
    """
    src = (CONTROLLER / "em_api.py").read_text()
    fn  = _fn_body(src, "_stream_file_to_device")
    code = "\n".join(l for l in fn.splitlines() if not l.lstrip().startswith("#"))

    assert "rm -f {dest}" not in code, (
        "deleting dest before sending destroys the rollback slot on any "
        "transfer that fails early"
    )
    # The .part cleanup on a bad md5 must survive — that one is load-bearing.
    assert "rm -f {landing}" in code or "rm -f {landing};" in code, (
        "a failed verification must still remove the .part"
    )


def test_a_failed_transfer_says_which_stage_it_failed_at():
    """
    One message covered five outcomes, and the two furthest apart are "the
    bytes arrived corrupt" and "no byte was ever sent". #121 was the second
    reported in the language of the first, and only the controller's own
    stdout could tell them apart — which is not something a user can produce
    mid-update.

    A device shell that answers nothing must NOT read as "no base64 decoder":
    one is a link problem worth retrying, the other is a property of the
    device that retrying cannot change.
    """
    src = (CONTROLLER / "em_api.py").read_text()

    assert "class TransferResult" in src and "__bool__" in src, (
        "the result must stay truthy so `if not await ...` call sites keep "
        "their meaning"
    )

    fn = _fn_body(src, "_stream_file_to_device")
    for stage in ("shell", "decoder", "send", "verify", "corrupt"):
        assert f'_transfer_failed("{stage}"' in fn, (
            f"the {stage} failure must be distinguishable from the others"
        )

    assert "if DETECT_MARKER not in detect_buf" in fn, (
        "a silent shell must report as a link problem, not a missing decoder"
    )

    # The OTA message must carry the stage through rather than re-flattening it.
    ota = src[src.index("_stream_binary_to_slot(live, binary"):]
    ota = ota[:ota.index("_monitor_reconnect")]
    assert "{ok}" in ota, (
        "the update failure message must name the stage the transfer reached"
    )
    # Comments stripped first, for the reason the free-space test gives: the
    # old wording is worth naming in a comment, and a test that reads its own
    # explanation as the bug can only be silenced by deleting the explanation.
    ota_code = "\n".join(l for l in ota.splitlines() if not l.lstrip().startswith("#"))
    assert "failed or did not verify" not in ota_code, (
        "the flattened message is what made #121 unreadable"
    )


def test_tested_firmware_build_matches_the_docs():
    """
    The wizard warns when a device is on a FireOS build other than the one
    EchoMuse is developed against, and docs/rooting.md tells people which to
    flash. Those two have to name the same build: a warning pointing at a
    version the docs do not mention is worse than no warning, because the
    person reading it has nowhere to go.

    Verified against the fleet 2026-08-07 — all three connected devices report
    ro.build.version.incremental = 272.6.8.0_user_680767620.
    """
    jsx = (CONTROLLER / "static" / "dashboard.jsx").read_text()
    m = re.search(r"_TESTED_FIREOS_BUILD\s*=\s*'([^']+)'", jsx)
    assert m, "dashboard.jsx no longer declares _TESTED_FIREOS_BUILD"
    build = m.group(1)

    rooting = (CONTROLLER.parent / "docs" / "rooting.md").read_text()
    assert build in rooting, (
        f"the wizard warns against build {build} but docs/rooting.md never "
        f"names it — a reader has nowhere to go"
    )


def test_push_log_event_callers_do_not_also_persist():
    """
    `_push_log_event` persists AND pushes. A caller that also calls
    `db.log_device` writes the line twice.

    That is not hypothetical: the device `log` handler did both, so every
    device log line landed twice about 6ms apart, and roughly half of
    `device_logs` was duplicates. It stayed invisible because a doubled log
    line looks like a device that logged twice.

    Two things it cost beyond the wasted rows. `em_support.thin_noise` keeps
    the newest three `[mem]` lines per device, so duplication halved the
    distinct readings a leak hunt gets from a bundle. And the redundant call
    was a synchronous SQLite write on the event loop, which
    `event_loop_lag_monitor` exists to catch.

    Checked by source shape because the alternative is importing
    em_controller, which the suite deliberately does not do.
    """
    root = Path(__file__).resolve().parent.parent
    for name in ("em_controller.py", "em_api.py"):
        lines = (root / name).read_text().splitlines()
        for i, line in enumerate(lines):
            if "_push_log_event(" not in line or "async def" in line:
                continue
            # The persist would sit just above the push, in the same block.
            window = lines[max(0, i - 6):i]
            offenders = [w.strip() for w in window if "db.log_device(" in w]
            assert not offenders, (
                f"{name}:{i+1} calls _push_log_event, which already persists, "
                f"but is preceded by {offenders[0]!r}. That writes the log "
                f"line twice. Drop the db.log_device call."
            )


# ── Ambient light status reaches a support bundle (#90) ──────────────────────
#
# Two users reported no light sensor and the bundle could not say why: the
# firmware knows whether the chip is absent or the driver simply has not
# bound, but writes that only to its own log, which the bundle does not
# collect and a reboot clears. Diagnosing it needed a shell session on their
# hardware. These pin the three links in the chain that fixes it, because
# each fails silently — a missing field just looks like an old device.

def test_register_handler_stores_ambient_light_status():
    """The controller must keep what the device reported at registration."""
    root = Path(__file__).resolve().parent.parent
    ctl = (root / "em_controller.py").read_text()
    assert 'device.ambient_light_status = msg.get("ambient_light_status")' in ctl, (
        "the register handler must store ambient_light_status off the register "
        "message — without it the reason is received and dropped"
    )


def test_bundle_live_state_carries_ambient_light_status():
    """And the support bundle must actually carry it.

    em_support takes live_state wholesale rather than through an allowlist, so
    the field has to be put there by em_api. A reason that never reaches a
    bundle leaves us exactly where #90 started.
    """
    root = Path(__file__).resolve().parent.parent
    api = (root / "em_api.py").read_text()
    assert '"ambient_light_status":' in api, (
        "em_api's live_state must include ambient_light_status so support "
        "bundles can answer why a device reports no light sensor"
    )
