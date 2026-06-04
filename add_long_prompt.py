"""
add_long_prompt.py - inserts one ~300-token prompt into the PROMPTS list
of modal_capture.py, without disturbing the rest of the file.
Run once:  python add_long_prompt.py
"""
import re, pathlib

LONG = (
    "Read the following passage carefully and then explain, in your own words, "
    "what it is describing. The water cycle is the continuous movement of water "
    "within the Earth and atmosphere. It begins when the sun heats water in "
    "oceans, lakes, and rivers, causing it to evaporate and rise into the air as "
    "water vapor. As this vapor rises, it cools and condenses into tiny droplets, "
    "forming clouds in a process called condensation. When the droplets in a "
    "cloud grow large and heavy enough, they fall back to the surface as "
    "precipitation, which can take the form of rain, snow, sleet, or hail. Some "
    "of this water soaks into the ground and is stored as groundwater, while "
    "some flows across the land as runoff, gradually making its way back into "
    "streams, rivers, and eventually the ocean. Plants also play a role: they "
    "absorb water through their roots and release it back into the air through "
    "their leaves in a process called transpiration. Together, evaporation, "
    "condensation, precipitation, runoff, and transpiration form a closed loop "
    "that recycles the same water over and over again across the entire planet. "
    "Because the total amount of water on Earth stays roughly constant, the water "
    "you drink today may have fallen as rain thousands of years ago, or even "
    "passed through a dinosaur long before humans existed. After reading this, "
    "summarize the five main stages of the water cycle and explain how they "
    "connect to one another in a single continuous process."
)

path = pathlib.Path("modal_capture.py")
src = path.read_text()
entry = '    "' + LONG.replace('"', '\\"') + '",\n]'
if "water cycle is the continuous" in src:
    print("long prompt already present, nothing to do."); raise SystemExit
src2 = re.sub(r"\n\]", "\n" + entry, src, count=1)
path.write_text(src2)
print("added a long (~300 token) prompt to PROMPTS in modal_capture.py")
