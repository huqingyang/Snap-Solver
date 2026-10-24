// Run with: node --test tests/test_followup_images.js
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function createApp(savedAutoSettings) {
    const storage = new Map(savedAutoSettings ? [['autoCaptureSettings', JSON.stringify(savedAutoSettings)]] : []);
    class Element {
        constructor() {
            this.value = '';
            this.dataset = {};
            this.style = {};
            this.children = [];
            this.listeners = {};
            this.classList = { add() {}, remove() {}, toggle() {} };
        }
        addEventListener(name, fn) { this.listeners[name] = fn; }
        querySelector() { return new Element(); }
        querySelectorAll() { return []; }
        append(...items) { this.children.push(...items); }
        appendChild(item) { this.children.push(item); }
        setAttribute(name, value) { this[name] = value; }
        getAttribute(name) { return this[name]; }
        scrollTo() {}
    }
    const elements = new Map();
    const element = id => {
        if (!elements.has(id)) elements.set(id, new Element());
        return elements.get(id);
    };
    const handlers = {};
    const emitted = [];
    const timers = new Map();
    let timerId = 0;
    let socketOptions;
    let now = 0;
    const socket = {
        connected: true,
        on(name, fn) { handlers[name] = fn; },
        emit(name, data) { emitted.push({ name, data }); },
    };
    const settings = {
        currentModel: { provider: 'alibaba', display_name: 'Qwen3.8 Flash' },
        currentTier: () => 'fast',
        hasKeyFor: () => true,
        addEventListener() {},
        maybeStartOnboarding() {},
        missingKeyForCurrentModel: () => null,
        getSettings: () => ({ model: 'qwen3.8-flash' }),
        collectApiKeys: () => ({}),
    };
    const context = vm.createContext({
        console,
        localStorage: { getItem: key => storage.get(key) || null, setItem: (key, value) => storage.set(key, value) },
        window: { settingsManager: settings, uiManager: { showToast() {} }, addEventListener() {} },
        document: {
            body: element('body'),
            getElementById: element,
            querySelector: element,
            createElement: () => new Element(),
            addEventListener() {},
        },
        io: options => { socketOptions = options; return socket; },
        TIER_INFO: { fast: { label: 'Fast' } },
        PROVIDER_TAB: { alibaba: '通义' },
        PROVIDER_LABEL: { alibaba: '阿里通义' },
        Sheets: { confirm: async () => 1 },
        requestAnimationFrame: () => 1,
        setTimeout(fn, delay) { timers.set(++timerId, { fn, delay, due: now + delay }); return timerId; },
        clearTimeout(id) { timers.delete(id); },
        Cropper: class {
            constructor(img) { this.image = img.src; }
            destroy() {}
            getCropBoxData() { return {}; }
            getCroppedCanvas() { return { toDataURL: () => this.image }; }
        },
    });
    vm.runInContext(fs.readFileSync('static/js/main.js', 'utf8') + '\nglobalThis.App = SnapSolver;', context);
    const app = new context.App();
    app.initialize();
    return {
        app, elements, emitted, timers, settings, handlers, socketOptions, storage,
        advance(ms) {
            const target = now + ms;
            let next;
            while ((next = [...timers].sort((a, b) => a[1].due - b[1].due).find(([, t]) => t.due <= target))) {
                const [id, timer] = next;
                now = timer.due;
                timers.delete(id);
                timer.fn();
            }
            now = target;
        },
        failScreenshot: () => handlers.screenshot_complete({ success: false, error: 'offline capture error' }),
        screenshot: image => handlers.screenshot_complete({ success: true, image }),
        complete: content => app.handleAiResponse({ status: 'completed', content }),
        request: () => emitted.filter(e => e.name === 'analyze_image').at(-1).data,
    };
}

test('screenshots after an answer extend the question and retain earlier follow-ups', async () => {
    const { app, elements, screenshot, complete, request } = createApp();
    screenshot('first');
    app.sendForSolve();
    complete('Original answer');
    app.followupInput.value = 'Explain the formula';
    app.sendFollowup();
    complete('Formula explanation');

    app.followupInput.value = 'Here is the rest';
    app.captureFollowup();
    screenshot('second');
    app.sendForSolve();
    assert.equal(app.mainAnswerText, 'Original answer');
    assert.equal(app.followupTurns.length, 2);
    assert.equal(app.followupTurns[0].answerText, 'Formula explanation');
    assert.deepEqual(Array.from(request().images), ['first', 'second']);
    assert.deepEqual(Array.from(request().history, t => t.content).slice(0, 3), [
        'Original answer', 'Explain the formula', 'Formula explanation',
    ]);
    assert.match(request().history.at(-1).content, /Here is the rest/);
    assert.match(request().history.at(-1).content, /第 2 至 2 张/);
    assert.equal(app.view, 'answer');
    complete('Answer using both images');

    // Subsequent text-only questions still carry every image.
    app.followupInput.value = 'Check again';
    app.sendFollowup();
    assert.deepEqual(Array.from(request().images), ['first', 'second']);
    complete('Checked');

    await elements.get('answerBack').listeners.click();
    assert.equal(app.view, 'empty');
    assert.equal(app.lastImages.length, 0);
    assert.equal(app.followupTurns.length, 0);
    screenshot('new-question');
    app.sendForSolve();
    assert.deepEqual(Array.from(request().images), ['new-question']);
    assert.equal(request().history, undefined);
});

test('canceling supplemental capture preserves the existing question', () => {
    const { app, screenshot, complete, request } = createApp();
    screenshot('first');
    app.sendForSolve();
    complete('Original answer');
    app.captureFollowup();
    screenshot('discarded');
    app.addNextImage();
    app.exitWorkspace();
    assert.equal(app.mainAnswerText, 'Original answer');
    assert.equal(app.pendingImages.length, 0);
    assert.deepEqual(Array.from(app.lastImages), ['data:image/png;base64,first']);
    app.followupInput.value = 'Continue';
    app.sendFollowup();
    assert.deepEqual(Array.from(request().images), ['first']);
});

const PROGRAM = '```python\nimport sys\ndef solve():\n    print(sum(map(int, sys.stdin.buffer.read().split())))\nif __name__ == "__main__":\n    solve()\n```';
const count = (emitted, name) => emitted.filter(e => e.name === name).length;

test('first auto screenshot is after exactly three seconds, ignoring the former interval', () => {
    const { app, advance, emitted, elements, screenshot, request } = createApp();
    assert.equal(app.startAutoCapture(600), true);
    assert.equal(elements.get('autoCaptureStartLabel').textContent, '停止自动截屏');
    advance(2999);
    assert.equal(count(emitted, 'capture_screenshot'), 0);
    advance(1);
    assert.equal(count(emitted, 'capture_screenshot'), 1);
    screenshot('first');
    assert.equal(count(emitted, 'analyze_image'), 1);
    assert.deepEqual(Array.from(request().images), ['first']);
    assert.equal(request().history, undefined);
});

test('long thinking and streaming never start another screenshot; five seconds starts at completion', () => {
    const { app, advance, emitted, screenshot, complete, timers } = createApp();
    app.startAutoCapture();
    advance(3000);
    screenshot('question');
    for (const status of ['started', 'thinking', 'thinking_complete', 'streaming']) {
        app.handleAiResponse({ status, content: '12. B, answer' });
        advance(60000);
        assert.equal(count(emitted, 'capture_screenshot'), 1);
        assert.equal(timers.size, 0);
        assert.equal(app.autoCaptureEnabled, true);
    }
    complete('12. B, answer');
    advance(4999);
    assert.equal(count(emitted, 'capture_screenshot'), 1);
    advance(1);
    assert.equal(count(emitted, 'capture_screenshot'), 2);
    assert.equal(count(emitted, 'stop_generation'), 0);
});

test('coding answers wait ten minutes; programming multiple-choice answers wait five seconds', () => {
    for (const [answer, delay] of [[PROGRAM, 600000], ['12. B, 时间复杂度为 O(n log n)', 5000],
                                 ['12. B, ```python\nprint(1)\n```', 5000]]) {
        const { app, advance, emitted, screenshot, complete } = createApp();
        app.startAutoCapture();
        advance(3000);
        screenshot('question');
        complete(answer);
        advance(delay - 1);
        assert.equal(count(emitted, 'capture_screenshot'), 1);
        advance(1);
        assert.equal(count(emitted, 'capture_screenshot'), 2);
    }
});

test('follow-ups pause a pending countdown and keep a coding question on the ten minute delay', () => {
    const { app, advance, emitted, screenshot, complete } = createApp();
    app.startAutoCapture();
    advance(3000);
    screenshot('coding');
    complete(PROGRAM);
    advance(599000);
    app.followupInput.value = '解释这个边界情况';
    app.sendFollowup();
    advance(60000);
    assert.equal(count(emitted, 'capture_screenshot'), 1);
    complete('边界情况说明');
    advance(599999);
    assert.equal(count(emitted, 'capture_screenshot'), 1);
    advance(1);
    assert.equal(count(emitted, 'capture_screenshot'), 2);
});

test('starting during an existing answer waits without interrupting it, then uses the answer delay', () => {
    const { app, advance, emitted, screenshot, complete } = createApp();
    screenshot('manual');
    app.sendForSolve();
    app.startAutoCapture();
    advance(3000);
    assert.equal(count(emitted, 'capture_screenshot'), 0);
    complete('12. B, answer');
    advance(5000);
    assert.equal(count(emitted, 'capture_screenshot'), 1);
    assert.equal(count(emitted, 'stop_generation'), 0);
});

test('a late automatic screenshot cannot interrupt thinking that began before it returned', () => {
    const { app, advance, emitted, screenshot, complete } = createApp();
    app.startAutoCapture();
    advance(3000);
    app.handleAiResponse({ status: 'thinking', content: 'working' });
    screenshot('too-late');
    assert.equal(app.generating, true);
    assert.equal(app.thinkingText, 'working');
    assert.equal(app.autoCaptureEnabled, true);
    assert.equal(count(emitted, 'analyze_image'), 0);
    assert.equal(count(emitted, 'stop_generation'), 0);
    complete('current answer');
    advance(4999);
    assert.equal(count(emitted, 'capture_screenshot'), 1);
    advance(1);
    screenshot('next');
    assert.equal(count(emitted, 'analyze_image'), 1);
});

test('drafts postpone capture and an automatic new question discards earlier follow-up history', () => {
    const { app, advance, emitted, screenshot, complete, request } = createApp();
    screenshot('old');
    app.sendForSolve();
    complete('old answer');
    app.followupInput.value = 'old followup';
    app.sendFollowup();
    complete('old explanation');
    app.startAutoCapture();
    app.followupInput.value = 'unsent draft';
    advance(3000);
    assert.equal(count(emitted, 'capture_screenshot'), 0);
    app.followupInput.value = '';
    advance(5000);
    screenshot('fresh');
    assert.deepEqual(Array.from(request().images), ['fresh']);
    assert.equal(request().history, undefined);
    assert.equal(app.followupTurns.length, 0);
});

test('stop, restart, capture failure, model error and empty answers leave no unexpected timers', () => {
    const { app, advance, emitted, screenshot, failScreenshot, timers, complete } = createApp();
    app.startAutoCapture();
    app.startAutoCapture();
    assert.equal(timers.size, 1);
    advance(3000);
    app.stopAutoCapture();
    screenshot('discard');
    assert.equal(count(emitted, 'analyze_image'), 0);
    assert.equal(timers.size, 0);
    app.startAutoCapture();
    advance(3000);
    failScreenshot();
    assert.equal(app.autoCaptureEnabled, false);
    app.startAutoCapture();
    app.handleAiResponse({ status: 'error', error: 'offline' });
    assert.equal(timers.size, 0);
    app.startAutoCapture();
    advance(3000);
    screenshot('empty-answer');
    complete('');
    assert.equal(app.autoCaptureEnabled, false);
    assert.equal(timers.size, 0);
});

test('home button can stop automation while the model is thinking and no countdown exists', () => {
    const { app, advance, elements, screenshot, timers } = createApp();
    const button = elements.get('autoCaptureStart');
    button.listeners.click();
    advance(3000);
    screenshot('question');
    assert.equal(timers.size, 0);
    assert.equal(button.getAttribute('aria-pressed'), 'true');
    button.listeners.click();
    assert.equal(app.autoCaptureEnabled, false);
    assert.equal(elements.get('autoCaptureStartLabel').textContent, '开始自动截屏');
});

test('phone disconnect cancels all scheduling and reconnection permits a fresh three-second start', () => {
    const { app, advance, emitted, handlers, socketOptions, timers } = createApp();
    assert.equal(socketOptions.reconnectionAttempts, Infinity);
    app.startAutoCapture();
    advance(3000);
    app.socket.connected = false;
    handlers.disconnect('ping timeout');
    assert.equal(app.captureInFlight, false);
    assert.equal(app.autoCaptureRequest, null);
    assert.equal(app.autoCaptureEnabled, false);
    assert.equal(timers.size, 0);
    app.socket.connected = true;
    handlers.connect();
    advance(1000000);
    assert.equal(count(emitted, 'capture_screenshot'), 1);
    app.startAutoCapture();
    advance(3000);
    assert.equal(count(emitted, 'capture_screenshot'), 2);
});


test('fixed mode checks every twenty seconds, skips busy ticks, and does not reset its clock at answer completion', () => {
    const { app, advance, emitted, screenshot, complete } = createApp();
    assert.equal(app.setAutoCaptureSettings({ fixedInterval: true, intervalSeconds: 20 }), true);
    app.startAutoCapture();
    advance(3000);
    screenshot('question');
    app.handleAiResponse({ status: 'thinking', content: 'working' });
    advance(45000); // Checks at 23s and 43s are skipped, with no model cancellation.
    assert.equal(count(emitted, 'capture_screenshot'), 1);
    complete(PROGRAM); // At 48s, a Coding answer must not switch fixed mode to a ten-minute delay.
    advance(14999);
    assert.equal(count(emitted, 'capture_screenshot'), 1);
    advance(1); // The next scheduled tick is at 63s, not 68s.
    assert.equal(count(emitted, 'capture_screenshot'), 2);
    assert.equal(count(emitted, 'stop_generation'), 0);
});

test('answer-based mode uses separately configurable normal seconds and coding minutes', () => {
    for (const [answer, delay] of [['12. B, answer', 7000], [PROGRAM, 120000]]) {
        const { app, advance, emitted, screenshot, complete } = createApp();
        app.setAutoCaptureSettings({ fixedInterval: false, normalSeconds: 7, codingMinutes: 2 });
        app.startAutoCapture();
        advance(3000);
        screenshot('question');
        complete(answer);
        advance(delay - 1);
        assert.equal(count(emitted, 'capture_screenshot'), 1);
        advance(1);
        assert.equal(count(emitted, 'capture_screenshot'), 2);
    }
});

test('capture settings persist without enabling automation and invalid edits keep the previous values', () => {
    const { app, storage, timers } = createApp();
    const config = { fixedInterval: true, intervalSeconds: 25, normalSeconds: 9, codingMinutes: 3 };
    assert.equal(app.setAutoCaptureSettings(config), true);
    assert.deepEqual(JSON.parse(storage.get('autoCaptureSettings')), config);
    const restored = createApp(JSON.parse(storage.get('autoCaptureSettings'))).app;
    assert.equal(restored.autoCaptureSettings.intervalSeconds, 25);
    assert.equal(restored.autoCaptureSettings.normalSeconds, 9);
    assert.equal(restored.autoCaptureSettings.codingMinutes, 3);
    assert.equal(restored.autoCaptureSettings.fixedInterval, true);
    assert.equal(restored.autoCaptureEnabled, false);
    for (const edit of [{ intervalSeconds: 0 }, { normalSeconds: 86401 }, { codingMinutes: 0.5 }, { intervalSeconds: 'bad' }]) {
        assert.equal(app.setAutoCaptureSettings(edit), false);
    }
    assert.deepEqual(JSON.parse(storage.get('autoCaptureSettings')), config);
    assert.equal(timers.size, 0);
});

test('switching capture modes while thinking keeps one timer at most and preserves first-shot delay', () => {
    const { app, advance, emitted, screenshot, complete, timers } = createApp();
    app.startAutoCapture();
    advance(2000);
    app.setAutoCaptureSettings({ fixedInterval: true, intervalSeconds: 20 });
    advance(1000);
    assert.equal(count(emitted, 'capture_screenshot'), 1);
    screenshot('question');
    app.handleAiResponse({ status: 'thinking', content: 'working' });
    assert.equal(timers.size, 1);
    app.setAutoCaptureSettings({ fixedInterval: false, normalSeconds: 9 });
    assert.equal(timers.size, 0);
    advance(60000);
    assert.equal(count(emitted, 'capture_screenshot'), 1);
    complete('12. B, answer');
    advance(8999);
    assert.equal(count(emitted, 'capture_screenshot'), 1);
    advance(1);
    assert.equal(count(emitted, 'capture_screenshot'), 2);
});
