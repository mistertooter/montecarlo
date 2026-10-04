"""Render a 3:4 Monte Carlo educational video with NumPy, Pillow and FFmpeg.
Install: pip install numpy pillow
Run: python monte_carlo.py
FFmpeg must be installed and on PATH. No market data is used.
"""
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS, SECONDS = 900, 1200, 30, 26
N, STEPS, YEARS = 5000, 120, 5
START, MU, SIGMA, SEED = 10000, .07, .20, 42
OUT = Path(__file__).resolve().parent
rng = np.random.default_rng(SEED)
dt = YEARS / STEPS
# Geometric Brownian motion: independent normal shocks for every path/time.
z = rng.standard_normal((N, STEPS))
increments = (MU - SIGMA**2 / 2) * dt + SIGMA * np.sqrt(dt) * z
paths = START * np.exp(np.column_stack((np.zeros(N), np.cumsum(increments, axis=1))))
q = np.percentile(paths, [5, 50, 95], axis=0)
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
fonts = {}
def font(size, bold=False):
    key = size, bold
    if key not in fonts:
        fonts[key] = ImageFont.truetype(BOLD if bold else FONT, size)
    return fonts[key]
def text(d, xy, value, size=20, color='#dce9f5', bold=False):
    d.text(xy, value, font=font(size, bold), fill=color)
def panel(d, box):
    d.rounded_rectangle(box, radius=20, fill='#101e30', outline='#294259', width=2)
def money(x):
    return f'${x:,.0f}'
def render(frame):
    t = frame / FPS
    # Brief introductory hold, 18-second simulation, then final summary.
    k = int(np.clip((t - 2) / 18, 0, 1) * STEPS)
    a = np.arange(H, dtype=np.float32)[:, None, None] / H
    bg = np.zeros((H, W, 3), dtype=np.uint8)
    bg[:] = (np.array([5, 12, 24]) * (1-a) + np.array([14, 29, 45])*a).astype('uint8')
    im = Image.fromarray(bg)
    d = ImageDraw.Draw(im)
    text(d, (42, 32), 'THE SCIENCE OF UNCERTAINTY', 18, '#5be4e7', True)
    text(d, (40, 62), 'MONTE CARLO', 49, bold=True)
    text(d, (42, 123), 'One starting point. Thousands of possible futures.', 23)
    for x, label, value in [(42,'SIMULATED PATHS','5,000'),(321,'STARTING VALUE','$10,000'),(600,'TIME HORIZON','5 years')]:
        panel(d,(x,172,x+258,250))
        text(d,(x+16,185),label,14,'#819cb5',True)
        text(d,(x+16,208),value,27,bold=True)
    panel(d,(32,271,868,663))
    text(d,(54,286),'01  RANDOM SHOCKS → POSSIBLE PATHS',20,'#7ce6ed',True)
    left, right, top, bottom = 114, 836, 340, 598
    maxval = 32000
    def pt(j, v):
        return (left + (right-left)*j/STEPS, bottom-(bottom-top)*float(v)/maxval)
    for v in [0,10000,20000,30000]:
        y=pt(0,v)[1]
        d.line((left,y,right,y),fill='#26394e')
        text(d,(48,y-10),f'{v/1000:.0f}k',17,'#819cb5')
    for yr in range(6):
        x=pt(yr*24,0)[0]
        d.line((x,top,x,bottom),fill='#1e3044')
        text(d,(x-17,607),f'{yr}y',16,'#819cb5')
    if k > 0:
        poly=[pt(j,q[0,j]) for j in range(k+1)]+[pt(j,q[2,j]) for j in range(k,-1,-1)]
        d.polygon(poly,fill='#173949')
        colors=['#345a79','#355478','#3d6269','#4e4978','#406f7a']
        for i in range(90):
            coords=[pt(j, np.clip(paths[i,j],0,maxval)) for j in range(k+1)]
            d.line(coords,fill=colors[i%5],width=1)
        for ix,col,width in [(0,'#f6b565',2),(2,'#9c8aff',2),(1,'#55efe3',4)]:
            d.line([pt(j,q[ix,j]) for j in range(k+1)],fill=col,width=width)
        x,y=pt(k,q[1,k]); d.ellipse((x-6,y-6,x+6,y+6),fill='#cafff8')
    text(d,(54,638),'90 sample paths • teal: median • shaded: 5th–95th percentiles',14,'#a4bbc9')
    panel(d,(32,680,868,934))
    text(d,(54,695),'02  DISTRIBUTION OF VALUES AT THIS TIME',20,'#7ce6ed',True)
    vals=paths[:,k]
    edges=np.linspace(0,40000,51)
    edges[-1]=np.inf  # Last bin includes all values above $39,200.
    counts, edges=np.histogram(vals,bins=edges)
    lx,rx,ty,by=90,834,738,879
    peak=max(counts.max(),1)
    for i,c in enumerate(counts):
        x=lx+i*(rx-lx)/50
        height=c/peak*(by-ty)
        d.rectangle((x,by-height,x+12,by),fill='#4e8ea9' if edges[i]<START else '#42c4bc')
    for v in [0,10000,20000,30000,40000]:
        x=lx+(rx-lx)*v/40000
        text(d,(x-23,889),(f'{v/1000:.0f}k' if v<40000 else '40k+'),15,'#819cb5')
    text(d,(56,917),'All 5,000 paths • dollars • last bin includes the upper tail',13,'#9bb3c9')
    for x,label,value,color in [(32,'MEDIAN VALUE',money(np.median(vals)),'#63e8db'),(317,'90% CENTRAL RANGE',f'{money(q[0,k])}–{money(q[2,k])}','#c2b5ff'),(602,'BELOW START',f'{np.mean(vals<START)*100:.1f}%','#ffbe80')]:
        panel(d,(x,950,x+266,1032))
        text(d,(x+14,964),label,14,'#819cb5',True)
        text(d,(x+14,991),value,19,color,True)
    if t<2: caption='Begin with $10,000. Sample a fresh shock at each time step.'
    elif t<9: caption='Each path follows the same rules, with different random shocks.'
    elif t<15: caption='The spread widens: uncertainty compounds over time.'
    elif t<20: caption='Percentiles summarize the range; the median tracks the center.'
    else: caption='5,000 futures estimate a distribution—not a guaranteed outcome.'
    text(d,(40,1050),caption,18,'#ffffff')
    text(d,(40,1083),'S(t+Δt) = S(t) × exp[(μ − σ²/2)Δt + σ√Δt × Z]',22,'#67dadd')
    text(d,(40,1120),'μ = 7%/year   σ = 20%/year   Z ~ N(0,1)   Δt = 1/24 year',17,'#a4bbc9')
    text(d,(40,1150),'Illustrative model • fixed assumptions • no fees or deposits • seed 42',15,'#819cb5')
    return im

def main():
    output=OUT/'monte_carlo_simulation.mp4'
    cmd=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
    p=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    for frame in range(FPS*SECONDS):
        im=render(frame)
        if frame==FPS*22: im.save(OUT/'monte_carlo_preview.png')
        p.stdin.write(im.tobytes())
        if frame%150==0: print(f'Rendered {frame}/{FPS*SECONDS}',flush=True)
    p.stdin.close()
    if p.wait()!=0: raise RuntimeError('FFmpeg encoding failed')
    print(output)
    print('Final median:',money(q[1,-1]),'Below start:',np.mean(paths[:,-1]<START))
if __name__=='__main__': main()
