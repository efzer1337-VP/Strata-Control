"""Read the installed setup's model catalog without executing setup or accessing the network."""
import ast
import re
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DESCRIPTIONS = {
    "qwen": "Универсальная модель. Q2_0 — меньше памяти и выше скорость; IQ2_XS — рекомендуемый компромисс; IQ3 — больше памяти ради качества.",
    "coder": "Для кода и инструментов. 256 из 512 экспертов. IQ1_M здесь хранит экспертов примерно как IQ3_S; это не обычная однобитная полная модель. Слабее в общих задачах и языках кроме английского.",
    "swift": "Swift 1.5: дообученная Qwen, которая меньше размышляет. Скорость одного токена и память близки к оригиналу той же квантовки. Нужен собственный mmproj Swift.",
    "unsloth": "Динамические квантовки: часть экспертов держится в RAM, остальные читаются из GGUF на SSD. UD-Q4_K_XL — экспериментальный вариант, в установщике только NVIDIA.",
}


@lru_cache(maxsize=1)
def catalog():
    tree = ast.parse((ROOT / "setup.py").read_text(encoding="utf-8-sig"))
    models, families = {}, {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        if "MODELS" in names:
            models = ast.literal_eval(node.value)
        if "FAMILIES" in names:
            for key, value in zip(node.value.keys, node.value.values):
                fields = {}
                for field, entry in zip(value.keys, value.values):
                    try:
                        fields[ast.literal_eval(field)] = ast.literal_eval(entry)
                    except (ValueError, TypeError):
                        pass  # URL-building expressions are not executed.
                families[ast.literal_eval(key)] = fields
    return models, families


def describe_file(filename):
    models, families = catalog()
    # Accept any shard of a published model, including alternative shard counts.
    normalized = re.sub(r"-\d{5}-of-\d{5}\.gguf$", ".gguf", Path(filename).name)
    for family, fam in families.items():
        for quant, detail in models.items():
            if family not in detail.get("families", ("qwen", "swift")):
                continue
            pattern = detail.get("file", fam["file"]).format(q=quant, i=1)
            if normalized == re.sub(r"-\d{5}-of-\d{5}\.gguf$", ".gguf", pattern):
                return {**detail, "family": family, "quant": quant, "title": fam["title"],
                        "mmproj": fam.get("mmproj", ""), "description": DESCRIPTIONS[family], "menu": True}
    if re.fullmatch(r"Qwen3\.8-Flash-Next-UD-Q6_K_XL(?:-\d{5}-of-\d{5})?\.gguf", Path(filename).name):
        return {"family": "unsloth", "quant": "UD-Q6_K_XL", "title": "Unsloth Q6 (ручная настройка)",
                "description": "Экспериментальный ручной workflow из docs/UNSLOTH_Q6.md. Не входит в меню setup; автоматически не импортируется.", "menu": False}
    if "Uncensored-IQ3_XXS" in Path(filename).name:
        return {"family": "orca", "quant": "IQ3_XXS", "title": "OrcaRouter (ручная настройка)",
                "description": "Нужны преобразование --compat-bf16 и собственный токенизатор по docs/ORCA.md. Автоматический импорт не включён.", "menu": False}
    return None


def first_shard(path):
    path = Path(path)
    return path.with_name(re.sub(r"-\d{5}-of-(\d{5})\.gguf$", r"-00001-of-\1.gguf", path.name))


def catalog_rows():
    models, families = catalog()
    for family, fam in families.items():
        for quant, detail in models.items():
            if family in detail.get("families", ("qwen", "swift")):
                yield (fam["title"], quant, str(detail["ram_gb"]), f"{detail['download_gb']:g}",
                       "Эксперимент" if detail.get("experimental") else "Штатный")
