import numpy as np

from projland.calibration import identity_calibration
from projland.markers import Marker
from projland.render import Glow, LetterLabel, Constellations, Renderer, Scene


def _marker(mid, x, y, s=50.0):
    corners = np.array(
        [[x, y], [x + s, y], [x + s, y + s], [x, y + s]], dtype=np.float32
    )
    return Marker(id=mid, corners=corners)


def test_letter_label_renders_for_known_id():
    cal = identity_calibration((640, 480))
    r = Renderer(projector_size=(640, 480))
    scene = Scene(effects=[LetterLabel()])
    img_with_letter = r.render(scene, [_marker(7, 200, 200)], cal)  # id 7 → 'a'
    img_without = r.render(scene, [_marker(0, 200, 200)], cal)       # id 0 not in map
    assert img_with_letter.sum() > 0
    assert img_without.sum() == 0


def test_constellations_respects_max_distance():
    cal = identity_calibration((640, 480))
    r = Renderer(projector_size=(640, 480))
    near = Scene(effects=[Constellations(max_distance_px=200)])
    far = Scene(effects=[Constellations(max_distance_px=50)])
    markers = [_marker(7, 100, 100), _marker(9, 200, 100)]  # ~125 apart
    img_near = r.render(near, markers, cal)
    img_far = r.render(far, markers, cal)
    assert img_near.sum() > img_far.sum()  # line drawn vs not


def test_glow_brightens_existing_pixels():
    cal = identity_calibration((400, 300))
    r = Renderer(projector_size=(400, 300))
    # Without glow
    plain = Scene(effects=[Constellations()])
    glow = Scene(effects=[Constellations(), Glow(weight=0.7)])
    markers = [_marker(7, 100, 100), _marker(9, 250, 200)]
    a = r.render(plain, markers, cal)
    b = r.render(glow, markers, cal)
    assert b.sum() > a.sum()
