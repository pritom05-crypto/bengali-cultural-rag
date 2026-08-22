import sys
from pathlib import Path
import pprint
import inspect

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"

sys.path.insert(0, str(SCRIPTS_DIR))

import v6_rag


print("=" * 80)
print("V6.23 RETRIEVE DEBUG")
print("=" * 80)

print("\nretrieve function:")
print(v6_rag.retrieve)

print("\nretrieve signature:")
print(inspect.signature(v6_rag.retrieve))

print("\nretrieve source:")
try:
    print(inspect.getsource(v6_rag.retrieve))
except Exception as e:
    print("Could not read source:", e)


query = "সে খুব অলস এবং কোনো কাজ করতে চায় না।"

print("\n" + "=" * 80)
print("TEST QUERY")
print("=" * 80)

print("Query:", query)

print("\nCalling retrieve()...")

try:
    raw = v6_rag.retrieve(query)

    print("\n" + "=" * 80)
    print("RAW RETURN VALUE")
    print("=" * 80)

    print("TYPE:")
    print(type(raw))

    print("\nLENGTH:")

    try:
        print(len(raw))
    except Exception:
        print("No len()")

    print("\nREPR:")
    pprint.pprint(
        raw,
        width=140,
        depth=8
    )

except Exception as e:

    print("\n" + "=" * 80)
    print("RETRIEVE ERROR")
    print("=" * 80)

    print(type(e).__name__)
    print(str(e))

print("\n" + "=" * 80)
print("AVAILABLE V6.23 OBJECTS")
print("=" * 80)

for name in dir(v6_rag):

    if not name.startswith("_"):

        obj = getattr(
            v6_rag,
            name
        )

        if callable(obj):

            print(
                f"{name:35} "
                f"{type(obj)}"
            )

print("\nDONE")