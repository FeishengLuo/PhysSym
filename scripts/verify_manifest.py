#!/usr/bin/env python3
"""Cross-platform verification of release-file SHA256 checksums."""
import hashlib
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = root / 'MANIFEST_SHA256.txt'
failures = []
count = 0
for line in manifest.read_text().splitlines():
    digest, relative = line.split('  ', 1)
    path = root / relative
    if not path.is_file():
        failures.append(f'missing: {relative}')
    elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        failures.append(f'digest mismatch: {relative}')
    count += 1
print(f'checked {count} release files')
if failures:
    print('\n'.join(failures))
    raise SystemExit(1)
print('OK: all checksums match')
