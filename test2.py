import re

raw_text = """
73'
J. Belluz
68'
2 - 0
M. B. Diop
61'
A. Fadal
D. Amadou
61'
A. Harris
K. Stewart-Baynes
61'
A. García
S. Bassett
49'
1 - 0
S. Wathuta
A. Rosa
46'
M. Pinto
W. Frederick
"""

lines = raw_text.strip().split('\n')

events = []
i = 0
while i < len(lines):
    if re.match(r"^\d{2}'$", lines[i]):
        time = lines[i]
        event = {"time": time, "details": []}
        i += 1
        while i < len(lines) and not re.match(r"^\d{2}'$", lines[i]):
            event["details"].append(lines[i])
            i += 1
        events.append(event)
    else:
        i += 1

# Display structured events
for e in events:
    print(f"{e['time']}")
    for detail in e['details']:
        print(f"  - {detail}")
