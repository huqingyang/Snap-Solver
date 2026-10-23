// Run with: node --test tests/test_followup_images.js
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function createApp() {
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
        missingKeyForCurrentModel: () => null,
        getSettings: () => ({ model: 'qwen3.8-flash' }),
        collectApiKeys: () => ({}),
    };
    const context = vm.createContext({
        console,
        window: { settingsManager: settings, uiManager: { showToast() {} }, addEventListener() {} },
        document: {
            body: element('body'),
            getElementById: element,
            querySelector: element,
            createElement: () => new Element(),
            addEventListener() {},
        },
        io: () => socket,
        TIER_INFO: { fast: { label: 'Fast' } },
        PROVIDER_TAB: { alibaba: '通义' },
        PROVIDER_LABEL: { alibaba: '阿里通义' },
        Sheets: { confirm: async () => 1 },
        requestAnimationFrame: () => 1,
        setInterval(fn, delay) { timers.set(++timerId, { fn, delay }); return timerId; },
        clearInterval(id) { timers.delete(id); },
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
        app, elements, emitted, timers, settings,
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

test('automatic capture starts a fresh full-screen question without old images or history', () => {
    const { app, emitted, screenshot, complete, request, timers } = createApp();
    screenshot('old');
    app.sendForSolve();
    complete('Old answer');
    app.followupInput.value = 'Old follow-up';
    app.sendFollowup();
    complete('Old explanation');
    assert.equal(app.startAutoCapture(15), true);
    assert.equal(timers.get(app.autoCaptureTimer).delay, 15000);
    const before = emitted.length;
    app.generating = true;
    app.runAutoCapture();
    assert.equal(emitted.length, before);
    app.generating = false;
    app.followupInput.value = 'Draft question';
    app.runAutoCapture();
    assert.equal(emitted.length, before);
    app.followupInput.value = '';
    app.runAutoCapture();
    assert.equal(emitted.at(-1).name, 'capture_screenshot');
    app.runAutoCapture();
    assert.equal(emitted.length, before + 1);
    screenshot('fresh-full-screen');
    assert.deepEqual(Array.from(request().images), ['fresh-full-screen']);
    assert.equal(request().history, undefined);
    assert.equal(app.followupTurns.length, 0);
    assert.equal(app.view, 'answer');
    app.stopAutoCapture();
    assert.equal(timers.size, 0);
});

test('turning automation off discards an in-flight capture and failures stop the timer', () => {
    const { app, emitted, screenshot, failScreenshot, timers } = createApp();
    assert.equal(app.startAutoCapture(5), true);
    app.runAutoCapture();
    app.stopAutoCapture();
    screenshot('discard-me');
    assert.equal(emitted.filter(e => e.name === 'analyze_image').length, 0);
    assert.equal(app.startAutoCapture(5), true);
    app.runAutoCapture();
    failScreenshot();
    assert.equal(timers.size, 0);
    assert.equal(app.autoCaptureTimer, null);
});

test('interval validation, repeated start and model errors leave no extra timers', () => {
    const { app, timers } = createApp();
    for (const seconds of [0, -1, 1.2, 'invalid', 86401]) {
        assert.equal(app.startAutoCapture(seconds), false);
    }
    assert.equal(app.startAutoCapture(10), true);
    assert.equal(app.startAutoCapture(20), true);
    assert.equal(timers.size, 1);
    app.handleAiResponse({ status: 'error', error: 'offline test' });
    assert.equal(timers.size, 0);
});
