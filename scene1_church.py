"""Scene 1 - sunset church, wedding car arrives. Fully procedural (numpy + Pillow + OpenCV -> ffmpeg).
Run:  python3 scene1_church.py   ->  scene1_church.mp4
"""
import math, random, subprocess
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFilter

W, H, S, FPS, DUR = 1280, 720, 2, 30, 6          # S = supersample for the painted background
BW, BH = W * S, H * S
rng = np.random.default_rng(7)
lerp = lambda a, b, t: a + (b - a) * t
ease_out = lambda t: 1 - (1 - t) ** 3
ease_io = lambda t: t * t * (3 - 2 * t)
P = lambda pts: [(x * S, y * S) for x, y in pts]


def vgrad(h, w, stops):
    ys = np.linspace(0, 1, h)
    out = np.zeros((h, w, 3), np.float32)
    for c in range(3):
        out[:, :, c] = np.interp(ys, [s[0] for s in stops], [s[1][c] for s in stops])[:, None]
    return out


def new_layer():
    return Image.new("RGBA", (BW, BH), (0, 0, 0, 0))


def leaves(d, cx, cy, sx, sy, n, seed, dark, rim):
    r = random.Random(seed)
    for _ in range(n):
        x = cx + max(-1.6 * sx, min(1.6 * sx, r.gauss(0, sx))); y = cy + max(-1.6 * sy, min(1.6 * sy, r.gauss(0, sy)))
        rx, ry = r.randint(16, 42), r.randint(12, 30)
        d.ellipse([(x - rx + 7) * S, (y - ry + 7) * S, (x + rx + 7) * S, (y + ry + 7) * S], fill=rim)
        d.ellipse([(x - rx) * S, (y - ry) * S, (x + rx) * S, (y + ry) * S], fill=dark)


# ---------------------------------------------------------------- painted background
def build_bg():
    yy, xx = np.mgrid[0:BH, 0:BW].astype(np.float32)
    sky = vgrad(BH, BW, [(0, (62, 40, 115)), (.35, (150, 70, 150)), (.6, (235, 110, 130)), (.75, (255, 175, 95)), (1, (255, 200, 120))])
    d = np.hypot(xx - 1000 * S, yy - 400 * S) / (520 * S)
    sky += (np.clip(1 - d, 0, 1) ** 2)[..., None] * np.array([120, 90, 20], np.float32)
    n = cv2.GaussianBlur(cv2.resize(rng.random((18, 40)).astype(np.float32), (BW, BH), interpolation=cv2.INTER_CUBIC), (0, 0), 25)
    cl = np.clip((n - .5) * 4, 0, 1) * np.clip(1 - yy / (BH * .6), 0, 1)
    sky = sky * (1 - cl[..., None] * .5) + cl[..., None] * np.array([255, 150, 150], np.float32) * .5
    img = Image.fromarray(np.clip(sky, 0, 255).astype(np.uint8)).convert("RGBA")

    # distant city (two haze layers)
    L = new_layer(); dr = ImageDraw.Draw(L); r = random.Random(3)
    for col, base, hmin, hmax, lit in [((150, 90, 160, 150), 470, 80, 230, False), ((110, 65, 140, 210), 485, 60, 170, True)]:
        x = -20
        while x < 1300:
            w, h = r.randint(40, 90), r.randint(hmin, hmax)
            dr.rectangle([x * S, (base - h) * S, (x + w) * S, base * S], fill=col)
            if lit:
                for _ in range(r.randint(0, 3)):
                    wx, wy = x + r.randint(6, w - 12), base - h + r.randint(10, h - 20)
                    dr.rectangle([wx * S, wy * S, (wx + 5) * S, (wy + 7) * S], fill=(255, 190, 120, 190))
            x += w + r.randint(-5, 10)
    img = Image.alpha_composite(img, L.filter(ImageFilter.GaussianBlur(2)))

    # rolling bushes behind church
    L = new_layer(); dr = ImageDraw.Draw(L)
    for i in range(30):
        dr.ellipse([(i * 45 - 40) * S, (470 + r.randint(-10, 10)) * S, (i * 45 + 70) * S, 560 * S], fill=(95, 55, 120, 255))
    img = Image.alpha_composite(img, L.filter(ImageFilter.GaussianBlur(1.5)))

    # ground
    a = np.array(img.convert("RGB")).astype(np.float32)
    a[520 * S:] = vgrad(BH - 520 * S, BW, [(0, (235, 130, 140)), (.4, (170, 80, 145)), (1, (70, 35, 95))])
    gy = np.clip(1 - np.hypot((xx - 700 * S) / (700 * S), (yy - 600 * S) / (110 * S)), 0, 1)[520 * S:]
    a[520 * S:] += gy[..., None] * np.array([70, 40, 10], np.float32)
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).convert("RGBA")

    # church
    L = new_layer(); dr = ImageDraw.Draw(L)
    lit, litTop, shade = (245, 190, 105), (255, 215, 130), (170, 95, 135)
    roofL, roofS = (222, 105, 75), (150, 60, 95)
    dr.rectangle(P([(330, 200), (432, 520)])[0] + P([(330, 200), (432, 520)])[1], fill=lit)          # tower
    dr.polygon(P([(432, 215), (446, 205), (446, 520), (432, 520)]), fill=shade)
    dr.rectangle([330 * S, 330 * S, 432 * S, 338 * S], fill=(225, 150, 95))
    dr.polygon(P([(320, 205), (381, 118), (446, 205)]), fill=roofL)
    dr.polygon(P([(381, 118), (446, 205), (381, 205)]), fill=roofS)
    dr.rectangle([378 * S, 92 * S, 384 * S, 122 * S], fill=litTop); dr.rectangle([369 * S, 102 * S, 393 * S, 108 * S], fill=litTop)
    dr.ellipse([361 * S, 235 * S, 401 * S, 275 * S], fill=(80, 35, 70)); dr.rectangle([361 * S, 255 * S, 401 * S, 300 * S], fill=(80, 35, 70))
    dr.rectangle([440 * S, 335 * S, 840 * S, 520 * S], fill=lit)                                      # nave
    dr.rectangle([824 * S, 335 * S, 840 * S, 520 * S], fill=(215, 150, 100)); dr.rectangle([440 * S, 335 * S, 456 * S, 520 * S], fill=(215, 150, 100))
    dr.polygon(P([(448, 335), (640, 238), (832, 335)]), fill=litTop)                                  # gable
    dr.polygon(P([(425, 340), (640, 215), (855, 340), (855, 352), (640, 230), (425, 352)]), fill=roofL)
    dr.polygon(P([(640, 215), (855, 340), (855, 352), (640, 230)]), fill=roofS)
    dr.rectangle([440 * S, 345 * S, 840 * S, 353 * S], fill=(200, 125, 95))
    dr.ellipse([622 * S, 265 * S, 658 * S, 305 * S], fill=(255, 225, 160)); dr.ellipse([634 * S, 276 * S, 646 * S, 296 * S], fill=(190, 100, 90))
    dr.ellipse([596 * S, 392 * S, 684 * S, 472 * S], fill=(230, 160, 100)); dr.rectangle([596 * S, 432 * S, 684 * S, 520 * S], fill=(230, 160, 100))
    dr.ellipse([606 * S, 402 * S, 674 * S, 462 * S], fill=(70, 30, 60)); dr.rectangle([606 * S, 432 * S, 674 * S, 520 * S], fill=(70, 30, 60))
    dr.rectangle([618 * S, 455 * S, 662 * S, 520 * S], fill=(200, 110, 90))
    for wx in (505, 775):
        dr.ellipse([(wx - 16) * S, 379 * S, (wx + 16) * S, 411 * S], fill=(255, 205, 115)); dr.rectangle([(wx - 16) * S, 395 * S, (wx + 16) * S, 460 * S], fill=(255, 205, 115))
        dr.rectangle([(wx - 16) * S, 395 * S, (wx + 16) * S, 460 * S], outline=(150, 70, 90), width=3 * S); dr.line(P([(wx, 385), (wx, 460)]), fill=(150, 70, 90), width=2 * S)
    for k in (2, 1, 0):
        y0 = 520 + k * 10
        dr.rectangle([(590 - k * 18) * S, y0 * S, (690 + k * 18) * S, (y0 + 10) * S], fill=[(235, 170, 120), (215, 140, 120), (190, 110, 125)][k])
    img = Image.alpha_composite(img, L.filter(ImageFilter.GaussianBlur(.8)))

    # foreground: trees with sun rim-light, fence, shrubs
    L = new_layer(); dr = ImageDraw.Draw(L)
    dark, rim = (58, 28, 84, 255), (255, 150, 100, 255)
    dr.line(P([(0, 95), (200, 70), (420, 18)]), fill=dark, width=26 * S)
    dr.line(P([(1280, 80), (1130, 60), (950, 10)]), fill=dark, width=22 * S)
    leaves(dr, 120, 20, 150, 55, 170, 1, dark, rim); leaves(dr, 1180, 10, 130, 50, 140, 2, dark, rim)
    leaves(dr, 100, 705, 120, 20, 80, 5, dark, rim); leaves(dr, 1200, 705, 140, 20, 90, 6, dark, rim)
    for k in range(7):
        x = 24 + k * 30
        dr.rectangle([x * S, 395 * S, (x + 8) * S, 545 * S], fill=(48, 24, 68)); dr.polygon(P([(x - 2, 395), (x + 4, 378), (x + 10, 395)]), fill=(48, 24, 68))
    for yb in (430, 495):
        dr.rectangle([20 * S, yb * S, 240 * S, (yb + 7) * S], fill=(48, 24, 68))
    img = Image.alpha_composite(img, L.filter(ImageFilter.GaussianBlur(1)))

    # painterly brush texture + horizon haze
    a = np.array(img.convert("RGB")).astype(np.float32)
    ch_ = np.zeros((BH, BW), np.float32)
    for gx, gy_, gr in [(505, 428, 40), (775, 428, 40), (640, 480, 55), (640, 285, 35)]:
        ch_ += np.clip(1 - np.hypot(xx - gx * S, yy - gy_ * S) / (gr * S), 0, 1) ** 2
    a += ch_[..., None] * np.array([110, 70, 10], np.float32)
    wall = np.clip((yy - 330 * S) / (190 * S), 0, 1)[..., None] * ((xx > 330 * S) & (xx < 860 * S) & (yy < 520 * S))[..., None]
    a *= 1 - wall * .22 * np.array([0, .5, 1], np.float32)
    tex = cv2.GaussianBlur(cv2.resize(rng.random((BH // 8, BW // 8)).astype(np.float32), (BW, BH), interpolation=cv2.INTER_CUBIC), (0, 0), sigmaX=12, sigmaY=2)
    tex = (tex - tex.mean()) / (tex.std() + 1e-6)
    a *= (1 + tex * .035)[..., None]
    band = np.exp(-((yy - 500 * S) / (70 * S)) ** 2)[..., None]
    a = a * (1 - band * .14) + band * .14 * np.array([255, 170, 150], np.float32)
    a = cv2.GaussianBlur(a, (0, 0), 1.0)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------- wedding car sprite
def car_sprite(angle, s):
    K, cw, ch, ox, oy = 4, 260, 170, 130, 130
    im = Image.new("RGBA", (cw * K, ch * K), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    q = lambda pts: [((ox + x) * K, (oy + y) * K) for x, y in pts]
    for k in range(5):
        d.ellipse([(ox - 100 - k * 6) * K, (oy - 9 - k * 2) * K, (ox + 100 + k * 6) * K, (oy + 9 + k * 2) * K], fill=(20, 5, 40, 28))
    for wx in (-52, 52):
        d.ellipse([(ox + wx - 21) * K, (oy - 35) * K, (ox + wx + 21) * K, (oy + 7) * K], fill=(30, 18, 45))
    d.rounded_rectangle([(ox - 85) * K, (oy - 42) * K, (ox + 85) * K, (oy - 10) * K], radius=14 * K, fill=(246, 222, 196))
    d.rounded_rectangle([(ox - 85) * K, (oy - 24) * K, (ox + 85) * K, (oy - 10) * K], radius=8 * K, fill=(205, 150, 150))
    d.polygon(q([(-58, -42), (-44, -72), (-20, -80), (30, -80), (48, -72), (64, -42)]), fill=(250, 230, 205))
    d.polygon(q([(-50, -46), (-38, -70), (-6, -70), (-6, -46)]), fill=(110, 70, 110)); d.polygon(q([(6, -46), (6, -70), (32, -70), (52, -46)]), fill=(110, 70, 110))
    d.polygon(q([(-44, -48), (-37, -64), (-30, -48)]), fill=(175, 135, 175))
    d.line(q([(-44, -72), (-20, -80), (30, -80), (48, -72)]), fill=(255, 205, 145, 255), width=2 * K)
    d.line(q([(-85, -42), (-70, -42)]), fill=(255, 205, 145, 255), width=2 * K)
    for i in range(12):
        x = -38 + i * 7.5
        d.ellipse([(ox + x - 5) * K, (oy - 85 - 3 * math.sin(i)) * K, (ox + x + 5) * K, (oy - 75 - 3 * math.sin(i)) * K], fill=[(255, 120, 150), (255, 235, 235), (255, 170, 60)][i % 3])
    d.line(q([(-40, -76), (-70, -60), (-92, -44)]), fill=(255, 130, 170), width=3 * K)
    d.ellipse([(ox - 91) * K, (oy - 36) * K, (ox - 78) * K, (oy - 24) * K], fill=(255, 240, 150))
    for wx in (-52, 52):
        d.ellipse([(ox + wx - 17) * K, (oy - 31) * K, (ox + wx + 17) * K, (oy + 3) * K], fill=(40, 25, 55))
        d.ellipse([(ox + wx - 7) * K, (oy - 21) * K, (ox + wx + 7) * K, (oy - 7) * K], fill=(225, 205, 205))
        for sp in range(4):
            a = angle + sp * math.pi / 2
            d.line([(ox + wx) * K, (oy - 14) * K, (ox + wx + 15 * math.cos(a)) * K, (oy - 14 + 15 * math.sin(a)) * K], fill=(110, 85, 110), width=2 * K)
    return im.resize((int(cw * S * s), int(ch * S * s)), Image.LANCZOS), int(ox * S * s), int(oy * S * s)


# ---------------------------------------------------------------- post-processing helpers
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
SUNX, SUNY = 860, 380
ang, dist = np.arctan2(yy - SUNY, xx - SUNX), np.hypot(xx - SUNX, yy - SUNY)
fall = np.clip(1 - dist / 900, 0, 1) ** 1.5
vig = (1 - .55 * np.clip(np.hypot((xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2)) / 1.25, 0, 1) ** 2.2)[..., None]
dust = [(random.random() * W, random.random() * H, random.uniform(-6, 6), random.uniform(-14, -4), random.choice([1, 1, 2, 3]), random.random() * 6) for _ in range(70)]


def post(frame, t):
    f = frame.copy()
    for k in range(5):                                                       # birds
        bx, by = 1350 - t * (70 + k * 8) + k * 70, 120 + k * 28 + math.sin(t * 2 + k) * 10
        wing = math.sin(t * 10 + k) * 5
        pts = np.array([(bx - 9, by - 3 + wing), (bx, by), (bx + 9, by - 3 + wing)], np.int32)
        cv2.polylines(f, [pts], False, (70, 35, 90), 2, cv2.LINE_AA)
    f = f.astype(np.float32)
    ray = (np.sin(ang * 6 + t * .25) * .5 + .5) * (np.sin(ang * 13 - t * .15 + 1.3) * .5 + .5)
    f += ((ray ** 2) * fall)[..., None] * np.array([255, 205, 130], np.float32) * (.13 + .03 * math.sin(t * 1.7))
    b = cv2.GaussianBlur(np.clip(f - 200, 0, None), (0, 0), 22); f += b * .9    # bloom
    lay = np.zeros((H, W, 3), np.uint8)                                      # floating dust motes
    for x0, y0, vx, vy, sz, ph in dust:
        cv2.circle(lay, (int((x0 + vx * t) % W), int((y0 + vy * t) % H)), sz, (255, 225, 170), -1, cv2.LINE_AA)
    f += cv2.GaussianBlur(lay, (0, 0), 2).astype(np.float32) * .7
    f *= np.array([1.03, .97, 1.04], np.float32)                             # colour grade
    n = np.clip(f / 255, 0, 1); n = n * n * (3 - 2 * n) * .35 + n * .65
    f = n * 255 * vig
    f += rng.normal(0, 5, (H, W, 1)).astype(np.float32)                      # film grain
    f *= min(1, t / .6) * min(1, (DUR - t) / .6)                             # fade in / out
    return np.clip(f, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- render
def main(out="scene1_church.mp4"):
    bg = build_bg()
    ff = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                           "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    for fr in range(FPS * DUR):
        t = fr / FPS
        e = ease_out(min(1, t / 4.8)); cx = lerp(1400, 640, e)
        cy = 548 + math.sin(t * 14) * 1.2 * (1 - e); s = lerp(.9, 1.05, e)
        spr, ax, ay = car_sprite((1400 - cx) / 18, s)
        frame = bg.copy(); frame.paste(spr, (int(cx * S - ax), int(cy * S - ay)), spr)
        z = 1 + .09 * ease_io(t / DUR)
        cw, ch = BW / z, BH / z
        x0, y0 = (BW - cw) / 2 + math.sin(t * .4) * 8, (BH - ch) * .55
        sc = W / cw
        M = np.float32([[sc, 0, -x0 * sc], [0, sc, -y0 * sc]])
        cam = cv2.warpAffine(np.asarray(frame), M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        ff.stdin.write(post(cam, t).tobytes())
    ff.stdin.close(); ff.wait()


if __name__ == "__main__":
    main()
