(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const node = (tag, text, className) => {
    const element = document.createElement(tag);
    if (text !== undefined) element.textContent = text;
    if (className) element.className = className;
    return element;
  };
  let state, profileId = null, selectedEngine = 'strata', profileConfig = {};
  let values = {}, dirty = false, revision = 0, online = false, switching = false;
  let polling, previewTimer, previewController, validationErrors = [], previewPending = false;
  let logActive = false, filterFavorites = false, fieldNodes = [], resolveUnsaved;
  const busyButtons = new Set();
  let stars = new Set();
  try { stars = new Set(JSON.parse(localStorage.getItem('control-favorites') || '[]')); } catch {}

  async function api(path, body, signal) {
    const response = await fetch('/api/' + path, {
      method: body === undefined ? 'GET' : 'POST', signal,
      headers: { 'X-Control-Token': window.CONTROL_TOKEN, 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body)
    });
    let data;
    try { data = await response.json(); } catch { throw Error('Control вернул неверный ответ. Повторите действие.'); }
    if (!response.ok || data.error) throw Error(data.error || 'Не удалось выполнить действие.');
    return data;
  }
  function put(id, text) { if ($(id).textContent !== String(text)) $(id).textContent = text; }
  function notify(text, error = false) {
    const dialog = document.querySelector('dialog[open]');
    let target = $('message');
    if (dialog) {
      target = dialog.querySelector('.dialog-message');
      if (!target) { target = node('div', '', 'dialog-message message'); dialog.prepend(target); }
    }
    target.textContent = text;
    target.hidden = false;
    target.classList.toggle('error', error);
  }
  function bindBusy(id, action) {
    $(id).onclick = () => runBusy($(id), action);
  }
  async function runBusy(button, action) {
    if (busyButtons.has(button)) return;
    busyButtons.add(button); button.disabled = true;
    try { await action(); } catch (error) { notify(error.message, true); }
    finally { busyButtons.delete(button); button.disabled = false; updateActions(); }
  }
  function currentValues() { return { ...values, profile_description: $('profile-description').value }; }
  function changed() {
    dirty = true; revision++;
    put('dirty-state', 'Есть несохранённые изменения'); put('save', 'Сохранить');
    previewPending = true; updateActions();
    renderPresets();
    clearTimeout(previewTimer); previewTimer = setTimeout(preview, 300);
  }
  function showValidation(errors) {
    validationErrors = errors;
    $('validation').hidden = !errors.length;
    $('validation').textContent = errors.length ? 'До запуска: ' + errors.join(' ') : '';
    updateActions();
  }
  async function preview() {
    if (!profileId) return;
    previewController?.abort(); previewController = new AbortController();
    const version = revision, id = profileId;
    try {
      const result = await api('preview', { profile_id: id, values: currentValues() }, previewController.signal);
      if (version !== revision || id !== profileId) return;
      put('command', result.command || 'Команда недоступна.');
      previewPending = false; showValidation(result.validation_errors || []);
    } catch (error) {
      if (error.name === 'AbortError' || version !== revision || id !== profileId) return;
      put('command', 'Проверьте параметры: ' + error.message);
      previewPending = false; showValidation([error.message]);
    }
  }
  function compatibility(engine) {
    if (engine === 'strata') return 'Strata: Qwen3.8 Flash Next, Coder, Swift и поддерживаемые Unsloth. Для каждой модели нужны собственные pack, токенизатор и совместимый mmproj. MTP использует подготовленный runtime.';
    if (engine === 'polystrata') return 'PolyStrata: нужен подготовленный JSON именно выбранной модели с её токенизатором и профилем экспертов. В этой установке есть конфигурация Ornith. Поля KV, окна MTP и vision относятся к адаптеру PolyStrata.';
    if (state?.engines?.[engine]?.kind === 'llama.cpp') return 'llama.cpp: локальный GGUF и совместимый mmproj для vision. GPU layers, KV, MoE и reasoning передаются этому серверу; поддержку определяют его версия и модель.';
    return 'Пользовательский EXE: укажите документированные аргументы сервера, по одному на строку. Шаблоны: {model}, {context}, {host}, {port}. {host} всегда означает 127.0.0.1. Другие поля применяются только через шаблон; RAM-сторож работает в Control.';
  }
  function renderFields(fields) {
    const fragment = document.createDocumentFragment();
    values = {}; fieldNodes = [];
    const sections = new Map();
    for (const field of fields) {
      const group = field.section || 'Параметры';
      if (!sections.has(group)) {
        const container = node('div');
        const heading = node('h3', group, 'section');
        container.append(heading); sections.set(group, container); fragment.append(container);
      }
      values[field.key] = field.value;
      const row = node('div', undefined, 'parameter');
      row.dataset.key = field.key;
      row.dataset.search = (field.label + ' ' + field.key + ' ' + field.description).toLocaleLowerCase();
      const star = node('button', stars.has(field.key) ? '★' : '☆', 'star');
      star.type = 'button';
      star.setAttribute('aria-label', 'Избранное: ' + field.label);
      star.setAttribute('aria-pressed', String(stars.has(field.key)));
      star.onclick = () => {
        if (stars.has(field.key)) stars.delete(field.key); else stars.add(field.key);
        try { localStorage.setItem('control-favorites', JSON.stringify([...stars])); } catch {}
        star.textContent = stars.has(field.key) ? '★' : '☆';
        star.setAttribute('aria-pressed', String(stars.has(field.key))); filterFields();
      };
      const caption = node('label'); caption.htmlFor = 'field-' + field.key;
      const hint = node('div', field.description, 'help'); hint.id = 'hint-' + field.key;
      caption.append(node('div', field.label, 'title'), hint);
      const holder = node('div', undefined, 'field-control');
      let input;
      if (field.type === 'bool') {
        input = node('input'); input.type = 'checkbox'; input.checked = Boolean(field.value);
        const label = node('label', undefined, 'bool-field'), text = node('span');
        text.textContent = input.checked ? 'Включено' : 'Выключено';
        label.append(input, text); holder.append(label);
        input.onchange = () => { values[field.key] = input.checked; text.textContent = input.checked ? 'Включено' : 'Выключено'; changed(); };
      } else if (field.choices) {
        input = node('select');
        const choices = [...field.choices];
        const choiceValue = choice => typeof choice === 'object' ? choice.value : choice;
        if (!choices.some(choice => String(choiceValue(choice)) === String(field.value))) choices.push(field.value);
        for (const choice of choices) { const option = node('option', typeof choice === 'object' ? choice.label : String(choice)); option.value = choiceValue(choice); input.append(option); }
        input.value = field.value; holder.append(input);
        input.onchange = () => { values[field.key] = input.value; changed(); };
      } else {
        const numeric = ['number', 'float'].includes(field.type);
        input = node(field.type === 'textarea' ? 'textarea' : 'input');
        if (input.tagName === 'INPUT') input.type = numeric ? 'number' : 'text';
        if (numeric) input.step = field.type === 'float' ? 'any' : '1';
        if (field.min !== undefined) input.min = field.min;
        if (field.max !== undefined) input.max = field.max;
        if (field.optional) { input.placeholder = field.default_hint || 'По умолчанию движка'; input.title = input.placeholder; }
        if (field.type === 'textarea' || field.type === 'path') row.classList.add('wide');
        input.value = field.value ?? '';
        input.oninput = () => { values[field.key] = numeric && !field.optional && input.value !== '' ? Number(input.value) : input.value; changed(); };
        holder.append(input);
        if (field.key === 'context') {
          holder.classList.add('slider-field');
          const slider = node('input'); slider.type = 'range'; slider.min = field.min ?? '4096';
          slider.max = Math.max(524288, Number(field.value) || 0); slider.step = 1024; slider.value = input.value;
          slider.setAttribute('aria-label', 'Контекст, токенов');
          slider.oninput = () => { input.value = slider.value; input.dispatchEvent(new Event('input')); };
          input.addEventListener('input', () => { slider.max = Math.max(524288, Number(input.value) || 0); slider.value = input.value; });
          holder.append(slider);
          const presets = node('select'); presets.setAttribute('aria-label', 'Пресет контекста');
          const placeholder = node('option', 'Выбрать пресет контекста'); placeholder.value = ''; presets.append(placeholder);
          for (const size of [8192,16384,32768,65536,131072,204800,262144,393216,524288]) {
            const option = node('option', size.toLocaleString('ru') + ' токенов'); option.value = size; presets.append(option);
          }
          presets.onchange = () => { if (presets.value) { input.value = presets.value; input.dispatchEvent(new Event('input')); presets.value = ''; } };
          holder.append(presets);
        }
        if (field.type === 'path') {
          const browse = node('button', 'Обзор…'); browse.type = 'button';
          browse.setAttribute('aria-label', 'Выбрать: ' + field.label);
          browse.onclick = () => runBusy(browse, async () => {
            const kind = field.key === 'poly_config' ? 'profile' : field.key === 'mtp' ? 'folder' : field.key === 'mmproj' ? 'mmproj' : 'model';
            const result = await api('browse', { kind });
            if (result.path) { input.value = result.path; input.dispatchEvent(new Event('input')); }
          });
          if (field.key === 'model' && selectedEngine === 'strata') {
            input.readOnly = true;
            holder.append(node('p', 'Для другой модели используйте «Добавить модель»: ей нужен собственный pack.', 'help'));
          } else holder.append(browse);
        }
      }
      input.id = 'field-' + field.key; input.setAttribute('aria-describedby', hint.id);
      row.append(star, caption, holder); sections.get(group).append(row); fieldNodes.push(row);
    }
    $('fields').replaceChildren(fragment); filterFields();
    $('select-mmproj').disabled = !fields.some(field => field.key === 'mmproj');
  }
  function filterFields() {
    const search = $('search').value.trim().toLocaleLowerCase(); let count = 0;
    for (const row of fieldNodes) {
      row.hidden = !row.dataset.search.includes(search) || (filterFavorites && !stars.has(row.dataset.key));
      if (!row.hidden) count++;
    }
    for (const section of $('fields').children) section.hidden = ![...section.querySelectorAll('.parameter')].some(row => !row.hidden);
    $('no-fields').hidden = Boolean(count) || !profileId;
  }
  function options(select, records, selected, emptyText) {
    const stamp = JSON.stringify(records);
    if (select.dataset.stamp !== stamp) {
      const children = [];
      if (emptyText) { const option = node('option', emptyText); option.value = ''; children.push(option); }
      for (const [id, name] of records) { const option = node('option', name); option.value = id; children.push(option); }
      select.replaceChildren(...children); select.dataset.stamp = stamp;
    }
    select.value = records.some(([id]) => id === selected) ? selected : '';
  }
  function renderSelectors() {
    if (!state) return;
    const profiles = state.profiles || [], engines = state.engines || {};
    for (const engine of ['strata', 'polystrata']) {
      $('engine-' + engine).setAttribute('aria-pressed', String(selectedEngine === engine));
      const available = engines[engine]?.installed;
      const saved = profiles.filter(profile => profile.engine === engine);
      const chosen = saved.find(profile => profile.id === profileId) || saved.find(profile => profile.id === state.engine_profiles?.[engine]) || saved[0];
      put(engine + '-install', available ? 'Установлен' : 'Требуются файлы установки');
      put(engine + '-profile', chosen?.name || 'Профилей пока нет');
      const running = (state.active_servers || []).find(server => server.engine === engine);
      put(engine + '-server', running ? 'Работает: ' + running.name : 'Сервер остановлен');
    }
    options($('engine-filter'), Object.entries(engines).filter(([id]) => !['strata','polystrata'].includes(id)).map(([id, engine]) => [id, engine.name + (engine.installed ? '' : ' · не установлен')]), selectedEngine, 'Выбрать другой движок');
    options($('new-engine'), Object.entries(engines).map(([id, engine]) => [id, engine.name]), $('new-engine').value || selectedEngine);
    const matching = profiles.filter(profile => profile.engine === selectedEngine);
    options($('profiles'), matching.map(profile => [profile.id, profile.name]), profileId || '', matching.length ? undefined : 'Нет профилей для этого движка');
    $('profiles').disabled = !matching.length || switching;
    $('memory-preset').hidden = selectedEngine !== 'strata';
    $('memory-preset').disabled = !profileId || !(state.presets || []).length;
    put('compatibility', compatibility(selectedEngine));
    renderPresets();
  }
  function renderPresets() {
    const available = profileId && state?.selected_profile?.id === profileId ? state.presets || [] : [];
    $('preset-panel').hidden = !available.length;
    if (!available.length) return;
    const selection = $('engine-preset').value;
    options($('engine-preset'), available.map(preset => [preset.id, preset.name]), available.some(preset => preset.id === selection) ? selection : available[0].id);
    const preset = available.find(item => item.id === $('engine-preset').value);
    put('preset-description', preset.description);
    const labels = Object.fromEntries((state.fields || []).map(field => [field.key, field.label]));
    $('preset-changes').replaceChildren(...Object.entries(preset.values).flatMap(([key, value]) => [
      node('dt', labels[key] || key), node('dd', (values[key] === '' || values[key] === undefined ? 'По умолчанию' : String(values[key])) + ' → ' + String(value))
    ]));
  }
  function loadProfile(data) {
    previewController?.abort(); clearTimeout(previewTimer); previewPending = false;
    profileId = data.selected_profile?.id || '';
    selectedEngine = data.selected_engine || data.selected_profile?.config?.launcher_engine || 'strata';
    profileConfig = data.selected_profile?.config || {};
    dirty = false; revision++;
    const profile = data.profiles?.find(item => item.id === profileId);
    $('profile-name').value = profile?.name || '';
    $('profile-description').value = profileConfig.profile_description || '';
    put('save', 'Сохранено'); put('dirty-state', profileId ? 'Изменений нет' : '');
    put('model-summary', profile?.model || 'Выберите профиль или добавьте локальную модель.');
    put('command', data.command || 'Создайте или импортируйте профиль для предпросмотра команды.');
    renderFields(data.fields || []);
    showValidation(data.validation_errors || []);
    $('empty-profile').hidden = Boolean(profileId);
    for (const id of ['profile-name','profile-description','save-as','export','delete-profile','save','preset-32k','preset-64k']) $(id).disabled = !profileId;
    renderSelectors();
  }
  const numeric = value => typeof value === 'number' && Number.isFinite(value);
  const format = value => numeric(value) ? value.toFixed(1) : 'Недоступно';
  function bar(id, value) { $(id).style.width = Math.max(0, Math.min(100, numeric(value) ? value : 0)) + '%'; }
  function selectedStatus() { return state?.status?.profile_id === profileId ? state.status : { checking: Boolean(profileId), running: false, ready: false }; }
  function updateActions() {
    const status = selectedStatus();
    const operating = (state?.tasks || []).some(task => ['running','queued'].includes(task.status) && task.serialized);
    const other = (state?.active_servers || []).some(server => server.id !== profileId);
    $('start').disabled = !online || !profileId || switching || operating || other || status.running || status.checking || previewPending || Boolean(validationErrors.length);
    $('stop').disabled = !online || !profileId || operating || !status.running;
    $('clear-memory').disabled = !online || (state?.tasks || []).some(task => task.action === 'Очистка RAM + VRAM' && ['running','queued'].includes(task.status));
    $('restart').disabled = !online || !profileId || operating || other || !status.running || previewPending || Boolean(validationErrors.length);
    $('web').disabled = !online || !status.ready;
    $('monitor').disabled = !online || !status.ready || !['strata','polystrata'].includes(selectedEngine);
    $('api-address').disabled = !profileId;
    for (const button of busyButtons) button.disabled = true;
  }
  function renderStatus() {
    const all = state.status || {}, status = selectedStatus();
    put('gpu-name', all.gpu_name || 'GPU: телеметрия недоступна');
    put('vram-value', numeric(all.vram_used_gib) && numeric(all.vram_total_gib) ? format(all.vram_used_gib) + ' / ' + format(all.vram_total_gib) + ' ГиБ' : 'Недоступно');
    bar('vram-bar', all.vram_total_gib ? all.vram_used_gib / all.vram_total_gib * 100 : null);
    const used = all.memory_total_gib - all.memory_available_gib;
    put('ram-value', numeric(used) ? format(used) + ' / ' + format(all.memory_total_gib) + ' ГиБ' : 'Недоступно');
    bar('ram-bar', all.memory_total_gib ? used / all.memory_total_gib * 100 : null);
    for (const name of ['gpu','cpu']) { const value = all[name + '_util']; put(name + '-value', numeric(value) ? format(value) + '%' : 'Недоступно'); bar(name + '-bar', value); }
    let text = !profileId ? 'Профиль не выбран' : status.checking ? 'Проверяем выбранный профиль…' : status.error ? 'Ошибка проверки: ' + status.error : status.ready ? 'Готов к работе' : status.running ? 'Процесс запущен, ожидаем готовность API' : status.phase === 'error' ? 'Сервер завершился с кодом ' + status.exit_code + '. Проверьте журнал.' : 'Остановлен';
    if (status.guard?.status === 'stopped_for_ram') text = 'Сервер остановлен RAM-сторожем. Освободите память или проверьте резерв.';
    if (status.ready && status.model_loaded === false) text = 'API готов · модель не загружена';
    put('server-title', (state.engines?.[selectedEngine]?.name || selectedEngine) + ' · сервер');
    put('server-status', online ? text : 'Нет связи с Control. Повторное подключение…');
    $('server-status').classList.toggle('ready', Boolean(status.ready && online));
    put('api-url', profileId ? 'http://127.0.0.1:' + port() + '/v1' : 'Адрес API появится после выбора профиля');
    const metrics = status.metrics || {}, engine = metrics.engine || {}, live = metrics.live || {}, totals = metrics.totals || {};
    const entries = [
      ['Контекст сервера', numeric(engine.max_context) ? engine.max_context.toLocaleString('ru') + ' токенов' : 'Недоступно'],
      ['Сейчас · генерация', numeric(live.tok_s) ? format(live.tok_s) + ' ток/с' : 'Нет данных'],
      ['Сейчас · очередь', numeric(live.queued) ? String(live.queued) : 'Нет данных'],
      ['Кэш экспертов · слоты', engine.expert_slots ?? 'Нет данных'],
      ['Сессия · входные токены', totals.prompt_tokens ?? 'Нет данных'],
      ['Сессия · выходные токены', totals.output_tokens ?? 'Нет данных']
    ];
    const stamp = JSON.stringify(entries);
    if ($('runtime-metrics').dataset.stamp !== stamp) {
      $('runtime-metrics').replaceChildren(...entries.flatMap(([label, value]) => [node('dt', label), node('dd', String(value))]));
      $('runtime-metrics').dataset.stamp = stamp;
    }
    put('guard-status', 'RAM-сторож: ' + (status.guard?.status === 'monitoring' && status.running ? 'работает' : status.guard?.status === 'stopped_for_ram' ? 'остановил сервер' : profileConfig.launcher_memory?.ram_reserve_gib ? 'включится при запуске из Control' : 'выключен'));
    const active = (state.active_servers || []).find(server => server.id !== profileId);
    const activeStamp = JSON.stringify(active);
    $('active-server').hidden = !active;
    if (active && $('active-server').dataset.stamp !== activeStamp) {
      const select = node('button', 'Открыть работающий профиль');
      select.onclick = () => selectProfile(active.id).catch(error => notify(error.message, true));
      const stop = node('button', 'Остановить «' + active.name + '»');
      stop.onclick = () => runBusy(stop, async () => { await api('stop', { profile_id: active.id }); notify('Остановка работающего профиля запрошена.'); await refresh(true); });
      $('active-server').replaceChildren(node('p', 'Работает ' + active.name + ' · порт ' + active.port + '. Выбор ' + (state.engines?.[selectedEngine]?.name || selectedEngine) + ' не меняет работающий сервер. Для нового запуска сначала остановите его.'), select, stop);
      $('active-server').dataset.stamp = activeStamp;
    }
    const taskStamp = JSON.stringify(state.tasks || []);
    if ($('tasks').dataset.stamp !== taskStamp) {
      const labels = { queued: 'В очереди', running: 'Выполняется', done: 'Готово', error: 'Ошибка' };
      const tasks = (state.tasks || []).slice(-6).reverse().map(task => node('div', (task.action || 'Операция') + ': ' + labels[task.status] + '. ' + (task.error || (task.status === 'done' ? (task.result?.message || '') : task.message)), 'task-item' + (task.status === 'error' ? ' error' : '')));
      $('tasks').replaceChildren(...(tasks.length ? tasks : [node('p', 'Операций пока нет', 'help')]));
      $('tasks').dataset.stamp = taskStamp;
    }
    if (logActive) put('log', all.log || 'Общий журнал Control пока пуст.');
    updateActions();
  }
  async function refresh(force = false) {
    if (!force && document.hidden) return;
    if (polling) { await polling; if (!force) return; }
    polling = (async () => {
      try {
        const data = await api('state'); state = data; online = true;
        if ($('message').textContent.startsWith('Нет связи с приложением управления:')) $('message').hidden = true;
        if (data.profile_errors?.length) notify('Некоторые профили не удалось прочитать: ' + data.profile_errors.join('; '), true);
        put('connection', 'Локальное управление · ' + data.app_version);
        if (!switching && (profileId === null || (!dirty && (profileId !== (data.selected_profile?.id || '') || selectedEngine !== data.selected_engine)))) loadProfile(data);
        else renderSelectors();
        renderStatus();
      } catch (error) {
        online = false; put('connection', 'Нет связи с Control');
        put('server-status', 'Не удалось обновить состояние. Проверьте, что Control запущен.');
        for (const metric of ['vram','ram','gpu','cpu']) { put(metric + '-value', 'Недоступно'); bar(metric + '-bar', 0); }
        $('runtime-metrics').replaceChildren(node('p', 'Нет актуальных данных: соединение с Control прервано.', 'help'));
        delete $('runtime-metrics').dataset.stamp;
        updateActions();
        notify('Нет связи с приложением управления: ' + error.message, true);
      }
    })();
    try { await polling; } finally { polling = null; }
  }
  async function saveCurrent() {
    const version = revision, id = profileId;
    await api('save', { profile_id: id, name: $('profile-name').value, values: currentValues() });
    if (version === revision && id === profileId) { dirty = false; put('dirty-state', 'Изменения сохранены'); put('save', 'Сохранено'); }
    await refresh(true);
    return !dirty;
  }
  async function mayLeave() {
    if (!dirty) return true;
    $('unsaved-dialog').returnValue = 'cancel';
    $('unsaved-dialog').showModal(); $('unsaved-cancel').focus();
    const choice = await new Promise(resolve => { resolveUnsaved = resolve; });
    if (choice === 'save') return saveCurrent();
    return choice === 'discard';
  }
  for (const choice of ['save','discard','cancel']) $('unsaved-' + choice).onclick = () => $('unsaved-dialog').close(choice);
  $('unsaved-dialog').addEventListener('close', () => { resolveUnsaved?.($('unsaved-dialog').returnValue || 'cancel'); resolveUnsaved = null; });
  async function switchSelection(path, body) {
    if (switching) return;
    switching = true;
    try {
      if (!await mayLeave()) return;
      await api(path, body); dirty = false; profileId = null;
    } finally { switching = false; renderSelectors(); await refresh(true); }
  }
  async function selectProfile(id) { if (id && id !== profileId) await switchSelection('select', { id }); }
  for (const engine of ['strata','polystrata']) bindBusy('engine-' + engine, () => selectedEngine === engine ? undefined : switchSelection('select-engine', { engine }));
  $('engine-filter').onchange = () => { const engine = $('engine-filter').value; if (engine) switchSelection('select-engine', { engine }).catch(error => notify(error.message, true)); };
  $('profiles').onchange = () => selectProfile($('profiles').value).catch(error => notify(error.message, true));
  $('profile-name').oninput = changed; $('profile-description').oninput = changed;
  $('search').oninput = filterFields;
  $('engine-preset').onchange = renderPresets;
  $('apply-engine-preset').onclick = () => {
    const preset = (state?.presets || []).find(item => item.id === $('engine-preset').value);
    if (!preset || state.selected_profile?.id !== profileId) return;
    for (const [key, value] of Object.entries(preset.values)) {
      const input = $('field-' + key); if (!input) continue;
      input.value = value;
      input.dispatchEvent(new Event(input.tagName === 'SELECT' ? 'change' : 'input'));
    }
    notify('Пресет «' + preset.name + '» применён к черновику. Сохраните профиль; для работающего сервера используйте явный перезапуск.');
  };
  $('favorites').onclick = () => { filterFavorites = !filterFavorites; $('favorites').setAttribute('aria-pressed', String(filterFavorites)); filterFields(); };
  bindBusy('save', async () => { await saveCurrent(); notify('Профиль сохранён. Работающий сервер использует прежние настройки до перезапуска.'); });
  $('save-as').onclick = () => { $('new-profile-name').value = $('profile-name').value + ' · копия'; $('name-dialog').showModal(); };
  function bindForm(id, action) { $(id).onsubmit = event => { event.preventDefault(); runBusy($(id).querySelector('[type=submit]'), action); }; }
  bindForm('name-form', async () => {
    const version = revision;
    await api('save-as', { profile_id: profileId, name: $('new-profile-name').value, values: currentValues() });
    if (version === revision) { dirty = false; profileId = null; }
    $('name-dialog').close(); await refresh(true); notify('Копия профиля сохранена.');
  });
  for (const [id, context] of [['preset-32k',32768],['preset-64k',65536]]) $(id).onclick = () => {
    const input = $('field-context'); if (!input) return;
    input.value = context; input.dispatchEvent(new Event('input'));
    notify('Контекст ' + context.toLocaleString('ru') + ' токенов. Сохраните профиль; для работающего сервера примените явный перезапуск.');
  };
  $('memory-preset').onclick = () => {
    $('engine-preset').value = 'memory'; renderPresets(); $('apply-engine-preset').click();
  };
  bindBusy('select-mmproj', async () => {
    const result = await api('browse', { kind: 'mmproj' }), input = $('field-mmproj');
    if (result.path && input) { input.value = result.path; input.dispatchEvent(new Event('input')); }
  });
  function updateNewEngine() {
    put('new-compatibility', compatibility($('new-engine').value));
    $('new-poly-label').hidden = $('new-engine').value !== 'polystrata';
  }
  bindBusy('new-model', async () => {
    if (!await mayLeave()) return;
    if (dirty) loadProfile(state);
    $('new-engine').value = selectedEngine;
    updateNewEngine(); $('model-dialog').showModal();
  });
  $('new-engine').onchange = updateNewEngine;
  bindForm('model-form', async () => {
    const result = await api('new', { engine: $('new-engine').value, model: $('new-model-path').value, name: $('new-model-name').value, poly_config: $('new-poly-config').value });
    if (!result.task_id) { dirty = false; profileId = null; }
    $('model-dialog').close(); await refresh(true);
    notify(result.task_id ? 'Подготовка модели началась. Прогресс и ошибки появятся в операциях Control и журнале.' : 'Модель добавлена. Проверьте совместимость и параметры до запуска.');
  });
  async function browse(kind, target) { const result = await api('browse', { kind }); if (result.path) $(target).value = result.path; }
  bindBusy('browse-model', () => browse('model', 'new-model-path'));
  bindBusy('browse-poly-config', () => browse('profile', 'new-poly-config'));
  bindBusy('browse-engine', () => browse('exe', 'engine-exe'));
  bindBusy('engines', async () => {
    $('engine-inventory').replaceChildren(...Object.values(state?.engines || {}).map(engine => node('p', engine.name + ': ' + (engine.installed ? 'установлен' : 'требуются файлы установки'), 'help')));
    $('engine-dialog').showModal();
    const result = await api('catalog');
    $('catalog').replaceChildren(...result.models.map(row => node('p', row[0] + ' · ' + row[1] + ' · RAM ориентир ' + row[2] + ' ГБ · GGUF около ' + row[3] + ' ГБ · ' + row[4])));
  });
  for (const variant of ['cuda','cpu']) bindBusy('install-' + variant, async () => {
    await api('install', { variant }); $('engine-dialog').close(); await refresh(true); notify('Установка движка началась. Прогресс появится в операциях Control.');
  });
  bindBusy('install-zip', async () => {
    const picked = await api('browse', { kind: 'zip' }); if (!picked.path) return;
    const name = prompt('Имя движка из ZIP', 'Мой движок'); if (!name?.trim()) return;
    await api('install-zip', { path: picked.path, name: name.trim(), kind: 'custom' });
    $('engine-dialog').close(); await refresh(true); notify('Архив устанавливается. Прогресс появится в операциях Control.');
  });
  bindForm('register-form', async () => {
    await api('register', { name: $('engine-name').value, exe: $('engine-exe').value });
    $('engine-dialog').close(); await refresh(true); notify('Движок подключён.');
  });
  bindBusy('clear-memory', async () => {
    await api('clear-memory', {});
    notify('Очистка памяти запрошена. Результат появится в операциях Control.');
    await refresh();
  });
  for (const action of ['start','restart','stop']) bindBusy(action, async () => {
    if (action !== 'stop' && dirty && !await saveCurrent()) return;
    await api(action, { profile_id: profileId, values: currentValues() });
    notify(({start:'Запуск',stop:'Остановка',restart:'Перезапуск'})[action] + ' запрошен. Состояние обновится автоматически.');
    await refresh(true);
  });
  function port() { const status = selectedStatus(); return (status.running ? status.port : values.port) || profileConfig.port || 8080; }
  async function copy(text) {
    try { await navigator.clipboard.writeText(text); notify('Скопировано.'); }
    catch { notify('Не удалось скопировать. Выделите текст и скопируйте вручную.', true); }
  }
  $('api-address').onclick = () => copy('http://127.0.0.1:' + port() + '/v1');
  $('web').onclick = () => window.open('http://127.0.0.1:' + port() + '/', '_blank', 'noopener');
  $('monitor').onclick = () => window.open('http://127.0.0.1:' + port() + '/monitor.html', '_blank', 'noopener');
  $('copy').onclick = () => copy(logActive ? $('log').textContent : $('command').textContent);
  function tab(log) {
    logActive = log; $('command').hidden = log; $('log').hidden = !log;
    for (const [id, active] of [['command-tab',!log],['log-tab',log]]) { $(id).setAttribute('aria-selected', String(active)); $(id).tabIndex = active ? 0 : -1; }
    put('console-title', log ? 'Журнал серверов' : 'Команда PowerShell');
    put('console-subtitle', log ? 'Общий журнал Control: записи всех запусков разделены именами файлов профилей.' : 'Предпросмотр текущих настроек. RAM-сторож включается при запуске из Control.');
    if (log) api('log').then(data => put('log', data.log || 'Журнал пока пуст.')).catch(error => notify(error.message, true));
  }
  $('command-tab').onclick = () => tab(false); $('log-tab').onclick = () => tab(true);
  for (const id of ['command-tab','log-tab']) $(id).onkeydown = event => {
    if (['ArrowLeft','ArrowRight','Home','End'].includes(event.key)) { event.preventDefault(); tab(event.key === 'End' || (event.key !== 'Home' && !logActive)); $(logActive ? 'log-tab' : 'command-tab').focus(); }
  };
  $('import').onclick = () => $('import-file').click();
  $('import-file').onchange = async () => {
    const file = $('import-file').files[0]; if (!file) return;
    try {
      if (file.size > 1024 * 1024) throw Error('Профиль больше допустимого 1 МиБ.');
      if (dirty && !await mayLeave()) return;
      await api('import', { config: JSON.parse(await file.text()) }); dirty = false; profileId = null;
      await refresh(true); notify('Профиль импортирован. Проверьте локальные пути и совместимость.');
    } catch (error) { notify(error.message, true); } finally { $('import-file').value = ''; }
  };
  bindBusy('export', async () => {
    const result = await api('export', { profile_id: profileId, values: currentValues() });
    result.config.launcher_profile_name = $('profile-name').value;
    const url = URL.createObjectURL(new Blob([JSON.stringify(result.config, null, 2)], { type: 'application/json' }));
    const link = node('a'); link.href = url; link.download = ($('profile-name').value || 'strata-profile').replace(/[\\/:*?"<>|]/g, '_') + '.json';
    link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  let deleteProfileId = '';
  $('delete-profile').onclick = async () => {
    if (!profileId || !await mayLeave()) return;
    deleteProfileId = profileId;
    $('delete-profile-name').textContent = state?.profiles?.find(profile => profile.id === deleteProfileId)?.name || 'Без имени';
    $('delete-dialog-message').hidden = true;
    $('delete-dialog').showModal();
    $('confirm-delete-profile').focus();
  };
  bindBusy('confirm-delete-profile', async () => {
    await api('delete-profile', { profile_id: deleteProfileId });
    dirty = false; profileId = null; deleteProfileId = '';
    $('delete-dialog').close();
    await refresh(true);
    notify('Профиль удалён. Файлы модели сохранены.');
  });
  document.querySelectorAll('[data-close]').forEach(button => { button.onclick = () => button.closest('dialog').close(); });
  document.addEventListener('visibilitychange', () => { if (!document.hidden) refresh(true); });
  window.addEventListener('beforeunload', event => { if (dirty) { event.preventDefault(); event.returnValue = ''; } });
  refresh(true);
  setInterval(() => refresh(), 3000);
})();
