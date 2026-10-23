# PolyStrata: параметры для Strata Control

Исследование от 23 октября 2026 года. Оно относится к установленному `Strata-data/polystrata/engine-poly/strata-poly.exe`, а не к Qwen3.8 engine `strata` из того же репозитория. Реальные модели не запускались, пользовательские профили не изменялись. Результаты не являются замером производительности этого ПК.

## Что проверено

- Локальный checkout PolyStrata: `e97c8aa47ca2c01025accdd32d07a01536d794ce`, origin `https://github.com/VecSzn/PolyStrata.git`.
- `Strata-data/polystrata/engine-poly/BUILD.json:1`: local build, версия `0.1.45`, CUDA toolkit 13, architecture 89, source fingerprint `60d3ab2538e52f2c`, vision CPU.
- Фактический `strata-poly.exe --help` прочитан без загрузки модели. Его флаги и значения по умолчанию совпадают с локальными `src/poly/main.cpp:1024` и `src/poly/engine.hpp:42`. `--help` возвращает код 2 согласно `main.cpp:1196`, это поведение parser, а не ошибка загрузки CUDA.
- Веб-проверка первичных источников: [parser в зафиксированном commit](https://github.com/VecSzn/PolyStrata/blob/e97c8aa47ca2c01025accdd32d07a01536d794ce/src/poly/main.cpp), [модели и ограничения backend](https://github.com/VecSzn/PolyStrata/blob/e97c8aa47ca2c01025accdd32d07a01536d794ce/docs/BACKENDS.md). Ниже выводы привязаны к полным локальным исходникам, в том числе ограничения, которые краткая справка не показывает.

Пути к исходникам ниже считаются от `Strata-data/polystrata/`, кроме явно названных файлов корневого Control. Номера строк фиксируют прочитанное состояние до изменений Control.

## Разделение движков и моделей

`docs/BACKENDS.md:11`, `src/poly/engine.cpp:985`:

| GGUF architecture | Native PolyStrata | Особенности |
| --- | --- | --- |
| `qwen35moe` | Qwen3.6, Ornith, Tiel Coder, MiniCPM-V | Q8 KV доступен при выполнении требования к размерности; наличие draft зависит от файла |
| `glm5-next`, `glm5next` | GLM 5.3 Flash | Автоматическое окно 4; проверять ограничения Windows из документации |
| `gemma4` | Gemma 4 | Для guesses нужен отдельный gemma4-assistant GGUF; KV только fp16 |
| `deepseek4` | DeepSeek V4 Flash | KV только fp16; документация сообщает guesses из повторов |
| `lfm2moe` | LFM2.5 | KV только fp16; документация сообщает guesses из повторов |
| `qwen4exp` | Не этот executable | Используется отдельный `strata`, с другим набором CLI |

Один только суффикс `.gguf` не подтверждает совместимость. `Engine::open()` откажет неподдерживаемой architecture (`engine.cpp:998`). Произвольная замена модели в base JSON также требует подходящего tokenizer и expert profile, которые текущий адаптер наследует из базовой конфигурации.

Нельзя переносить в Poly поля Strata `--spec`, `--prefill`, `--expert-cache`, `--kv int8`. У Poly другие флаги. Встроенные аппаратные ограничения следует читать в `docs/BACKENDS.md:1986`; наличие source HIP не означает, что локальный CUDA EXE поддерживает AMD.

## Существующие поля Control: расхождения

Источник: корневой `strata_control_engines.py:323` до изменения и реальный `strata-ornith-uncensored-q3.json:1`.

1. `poly_threads` действительно управляет `--threads`, то есть потоками CPU для экспертов вне VRAM. Native default 0 означает все разрешённые процессу CPU (`engine.hpp:45`, `engine.cpp:1776`). Нынешний диапазон Control 1..256 исключает auto. Значение базового пользовательского JSON 16 является настройкой установки, не native default.
2. `poly_window` действительно управляет `--window`. Это максимум токенов известного продолжения и guesses, а не размер контекста. Native диапазон **0..8**, где 0 означает auto, 1 отключает guesses. Control 1..16 ошибочно разрешает 9..16 и не разрешает auto. Native выберет 8 либо 4 в зависимости от архитектуры (`engine.cpp:1638`, `1735`). Базовый JSON явно задаёт 8.
3. `poly_kv`: parser принимает именно `fp16` и `q8_0` (`main.cpp:1150`). В `Engine::open()` Q8 разрешён только для `qwen35moe` (`engine.cpp:990`), затем `qwen35.cpp:54` требует `(n_kv * head_dim) % 32 == 0`. Для неизвестной архитектуры нельзя обещать Q8. Native default fp16.
4. `top_k` в `sampling` JSON этого server должен быть целым **1..64** (`serve/server.py:7251`). Нынешний адаптер принимает 0..1000 и создаёт JSON, с которым server откажется стартовать. Native CLI sampler сам шире (`main.cpp:82`, `103`), но Control использует Python server, поэтому нужно соблюдать именно server contract.
5. Native context минимум 16 (`engine.cpp:1631`), native default 8192. Ограничение Control 16384..1048576 является политикой панели, а не пределами движка; текущее значение пользовательского JSON 32768. Верхний предел контекста нельзя считать обещанием, что память под него поместится.
6. `vram_reserve_gib` переводится в MiB. Значение 0 не оставит строго ноль: native зажимает явный резерв минимум до 256 MiB (`engine.cpp:1157`). Отрицательный native default означает автоматический резерв, текущая панель не предлагает его.
7. Базовый config задаёт `--prompt-cache 2`, хотя native default 6. Это следует показать как унаследованное значение, не заменить молча.
8. Текущий адаптер полностью заменяет `sampling` на temperature/top_p/top_k. При добавлении полей следует сливать явные overrides с базовым sampling, чтобы не терять существующие penalty/seed/min_p.

## Настройки, которые можно открыть без правок native backend

Пустое необязательное поле должно сохранять базовый JSON или не добавлять флаг. Native default и default конкретного профиля нужно различать. Указанные далее разумные верхние границы интерфейса, если они вводятся дополнительно, не следует описывать как native hard limit.

| Поле / CLI | Native default | Подтверждённый диапазон и смысл | Первичный источник |
| --- | --- | --- | --- |
| Потоки экспертов / `--threads` | 0 auto | 0 auto либо положительное число; имеет смысл подбирать под CPU | `engine.hpp:45`, `engine.cpp:1776` |
| Пакет prompt / `--prompt-batch` | 1024 | 1..4096 токенов; расходует scratch VRAM, не переносит VRAM в RAM | `engine.cpp:1632`, `main.cpp:1061` |
| Staging VRAM / `--stage-mib` | 2048 | >=1 MiB; фактически ограничен размером экспертов слоя | `engine.cpp:1634`, `1936` |
| Порог копирования заранее / `--stage-ahead` | 1024 | >=0; 0 выключает | `engine.cpp:1635`, `2401` |
| Staging длинного prompt / `--stage-read-mib` | 0 | 0 выключает; положительное число MiB; используется с VMM без layer split | `engine.hpp:56`, `engine.cpp:1940` |
| Доля CPU / `--cpu-share` | -1 auto | Отрицательное = адаптивно; 0 выключено; положительное зажимается до 1; UI разумно -1 либо 0..1 | `engine.hpp:65`, `engine.cpp:2500`, `2670` |
| Перестановки экспертов / `--expert-swaps` | 64 | 0..64; 0 фиксирует начальный cache, не делает весь процесс детерминированным | `engine.cpp:1636`, `main.cpp:1058` |
| Окно guesses / `--window` | 0 auto | 0..8; 1 без guesses | `engine.cpp:1638`, `1735` |
| Поиск повторов / `--lookup` | 3 | 0 выключает; положительное количество совпадающих токенов | `main.cpp:1045`, `1128`, `1160` |
| Отключить draft / `--no-draft` | flag отсутствует | Только выключает draft block, lookup управляется отдельно | `engine.hpp:77`, `main.cpp:1161` |
| KV целиком / `--kv-whole` | flag отсутствует | Выделить KV под весь context сразу; увеличивает начальное потребление VRAM | `engine.hpp:72`, `main.cpp:1054` |
| Checkpoints / `--prompt-cache` | 6 | >=0; 0 выключает; установленный base JSON использует 2 | `conversations.hpp:30`, `main.cpp:1217` |
| Интервал checkpoints / `--prompt-cache-every` | 2048 | >=0 токенов; 0 без промежуточных checkpoints | `conversations.hpp:31`, `main.cpp:1217` |
| Кэш бесед / `--conversation-cache-mib` | 0 | 0..16777216 MiB native; 0 выключает, панель может ограничить по RAM | `main.cpp:1216` |
| Число бесед / `--conversation-cache-slots` | 4 | 0..1024 | `conversations.hpp:34`, `main.cpp:1218` |
| Минимум беседы / `--conversation-cache-min-tokens` | 1024 | >=0 токенов | `conversations.hpp:35`, `main.cpp:1218` |
| Свободная RAM для кэша / `--conversation-cache-min-free-mib` | 2560 | 0..16777216 MiB native; является условием парковки беседы | `main.cpp:324`, `1219` |
| Параллельные запросы / top-level `parallel` | absent/1 | Для Control 1..8; server преобразует 2..8 в `--batch N`; уже существующий batch имеет приоритет | `serve/server.py:2718`, `main.cpp:1206` |

Для первого дополнения панели приоритетны prompt batch, staging MiB, checkpoints, интервал checkpoints, expert swaps, lookup, draft и KV allocation. Stage-ahead, stage-read и CPU share подходят для расширенных параметров.

`--no-graphs`, `--no-gpu-experts`, `--any-node` тоже существуют, но нужны главным образом для диагностики и специфичного CPU/NUMA. `--tail-skip` и `--skip-weight` пропускают вычисления экспертов и меняют ответы (`engine.cpp:1640`, `main.cpp:1096`), поэтому их нельзя включать в обычный стартовый preset без явного выбора пользователя. `STRATA_GDN_MMA=1` экспериментален, off по умолчанию и CUDA/shape-зависим (`docs/BACKENDS.md:29`), его нельзя выдавать за универсальную настройку.

## Sampling и reasoning: это ключи JSON, не CLI запуска

Server отправляет sampling через GEN/GENI protocol (`serve/server.py:1192`, `src/poly/main.cpp:563`). Флаги CLI `--temperature`, `--top-p`, `--min-p`, `--seed` из help относятся к запуску `--tokens`. Для Control параметры должны попасть в верхнеуровневый блок `sampling` runtime JSON.

| Ключ sampling | Допустимое значение server config | Значение выключения |
| --- | --- | --- |
| `temperature` | finite number >=0 | 0 greedy |
| `top_p` | 0 < x <=1 | 1 |
| `top_k` | integer 1..64 | Не использовать 0 в JSON |
| `min_p` | 0..1 | 0 |
| `presence_penalty` | number >=0 | 0 |
| `frequency_penalty` | number >=0 | 0 |
| `repetition_penalty` | number >0 | 1 |
| `penalty_last_n` | integer >=0 | При активной penalty server использует 64, если поле не положительное |
| `seed` | positive integer либо отсутствует | Отсутствует; UI 0 auto должен удалять ключ |

Источники диапазонов: `serve/server.py:7238`..`7274`; применение penalties/window: `1192`..`1228`; native sampler: `src/poly/main.cpp:79`..`156`. Валидация Control дополнительно должна отклонять NaN/Infinity. Default установленного JSON: temperature 1, top_p .95, top_k 20; модельные fallback значения server зависят от architecture и thinking (`server.py:7523`). Отсутствующее поле не равно нулю.

Поля верхнего уровня runtime JSON:

- `reasoning_effort`: `none`, `low`, `medium`, `high`; `none` отключает thinking. Не передавать как нативный флаг. Источники `serve/server.py:7108`, `7306`, `4054`.
- `reasoning_budget_tokens`: целое; 0 означает без лимита, положительное ограничивает thinking при включённом thinking. Это не общий `max_tokens` ответа. Источники `serve/server.py:3165`, `3725`, `7314`.
- `parallel`: 1..8 по server/native contract; расходует отдельные state и KV для последовательностей. Не стоит автоматически повышать.

Запрос клиента имеет приоритет над defaults; shared settings web chat также имеют приоритет над config sampling (`serve/server.py:4694`). Следовательно, поля панели задают значения по умолчанию, а не принудительную политику приложений.

## Три opt-in стартовых сценария

Это проектные стартовые значения для ручного выбора, не доказанные оптимальные настройки. Они не применяются автоматически к существующим профилям, не меняют модель, tokenizer, порт, vision или его файлы. Все они сохраняют fp16 KV, обычное вычисление всех выбранных экспертов и GPU graphs. На ПК после выбора требуется обычная проверка свободной RAM/VRAM.

| Параметр | Повседневный | Меньше памяти | Несколько бесед |
| --- | --- | --- | --- |
| Context | 32768 | 16384 | 65536 |
| CPU threads | 0 auto | 0 auto | 0 auto |
| KV | fp16 | fp16 | fp16 |
| Window | 0 auto | 0 auto | 0 auto |
| Prompt batch | 1024 | 256 | 1024 |
| Stage MiB | 2048 | 512 | 2048 |
| Prompt checkpoints | 2 | 1 | 6 |
| Checkpoint every | 2048 | 4096 | 2048 |
| Conversation cache MiB | 0 | 0 | 2048 |
| Cache slots | 4 | 4 | 4 |
| Minimum free RAM for parking MiB | 2560 | 2560 | 4096 |
| Expert swaps | 64 | 64 | 64 |
| Lookup | 3 | 3 | 3 |
| Parallel | 1 | 1 | 1 |

Снижение context, prompt batch, staging и числа checkpoints уменьшает явно запрошенные объёмы соответствующих буферов; насколько меняются общий расход и задержка, требует замера на модели и ПК. Кэш бесед использует RAM для парковки; его 2048 MiB не гарантируют, что четыре длинные беседы поместятся. Preset «Несколько бесед» означает чередование диалогов, не конкурентную генерацию.

Sampling следует сохранять из профиля, а не заменять универсальными числами для всех семейств. Допустимо отдельным явным сценарием «Код» предложить temperature .6, top_p .95, top_k20 и presence_penalty0 **только для Qwen3.6**, как отмечает `serve/server.py:7527`; архитектура `qwen35moe` сама по себе не доказывает применимость model card Qwen к Ornith/Tiel fine-tune. Аналогично reasoning `low`/`high` задаёт запрос шаблону и не означает заранее измеренного качества или времени ответа.

## Проверка реализации

Без реальной модели можно проверить preview runtime JSON: ровно один каждого управляемого флага; absent optional сохраняет base; 0 auto для threads/window сохраняется; window9, top_k0/65, negative caches и неверный KV отклоняются; новые поля round-trip через профиль; обе кнопки пресета меняют только выбранный движок; preview не пишет runtime файл. Отдельно требуется case с базовым sampling `min_p`/penalty, чтобы подтвердить сохранение.

Фактическую совместимость Q8 с architecture можно подтвердить прочтением GGUF metadata без загрузки весов. Изменять нативные CUDA/HIP/SYCL файлы для перечисленных полей не требуется. Реальные генерация, производительность, multi-GPU и GPU vision в этом исследовании не проверялись.
