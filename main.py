"""
Evacuación en Mina Subterránea - BFS/UCS/Greedy/A*
Taller IA 2026
"""
from collections import deque
import heapq, webbrowser, os, json, time

COSTS = {'.': 1, 'S': 1, 'E': 1, 'F': 1, 'G': 10, '#': float('inf'), 'X': float('inf')}

# ==================== MAPA Y ALGORITMOS ====================
class Mina:
    def __init__(self, txt):
        self.g = [list(l) for l in txt.strip().split('\n')]
        self.h, self.w = len(self.g), len(self.g[0])
    
    def vecinos(self, p):
        r, c = p
        return [(r+dr, c+dc) for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                if 0 <= r+dr < self.h and 0 <= c+dc < self.w and self.g[r+dr][c+dc] not in '#X']
    
    def costo(self, p): return COSTS.get(self.g[p[0]][p[1]], 1)
    def encontrar(self, chars): return [(r,c) for r in range(self.h) for c in range(self.w) if self.g[r][c] in chars]
    def clonar(self): m = Mina.__new__(Mina); m.g, m.h, m.w = [r[:] for r in self.g], self.h, self.w; return m

def reconstruir(cf, ini, fin):
    p, c = [], fin
    while c: p.append(c); c = cf.get(c)
    return p[::-1]

def bfs(m, ini, metas):
    q, cf, exp = deque([ini]), {ini: None}, 0
    while q:
        c = q.popleft(); exp += 1
        if c in metas: return reconstruir(cf, ini, c), sum(m.costo(p) for p in reconstruir(cf, ini, c)[1:]), exp
        for v in m.vecinos(c):
            if v not in cf: q.append(v); cf[v] = c
    return [], float('inf'), exp

def ucs(m, ini, metas):
    q, cf, cs, exp, n = [(0,0,ini)], {ini: None}, {ini: 0}, 0, 0
    while q:
        _, _, c = heapq.heappop(q); exp += 1
        if c in metas: return reconstruir(cf, ini, c), cs[c], exp
        for v in m.vecinos(c):
            nc = cs[c] + m.costo(v)
            if v not in cs or nc < cs[v]: cs[v] = nc; n += 1; heapq.heappush(q, (nc, n, v)); cf[v] = c
    return [], float('inf'), exp

def h(p, metas): return min(abs(p[0]-g[0]) + abs(p[1]-g[1]) for g in metas) if metas else 0

def greedy(m, ini, metas):
    q, cf, vis, exp, n = [(h(ini,metas),0,ini)], {ini: None}, set(), 0, 0
    while q:
        _, _, c = heapq.heappop(q)
        if c in vis: continue
        vis.add(c); exp += 1
        if c in metas: return reconstruir(cf, ini, c), sum(m.costo(p) for p in reconstruir(cf, ini, c)[1:]), exp
        for v in m.vecinos(c):
            if v not in vis: n += 1; heapq.heappush(q, (h(v,metas), n, v)); cf.setdefault(v, c)
    return [], float('inf'), exp

def astar(m, ini, metas):
    q, cf, cs, exp, n = [(h(ini,metas),0,ini)], {ini: None}, {ini: 0}, 0, 0
    while q:
        _, _, c = heapq.heappop(q); exp += 1
        if c in metas: return reconstruir(cf, ini, c), cs[c], exp
        for v in m.vecinos(c):
            nc = cs[c] + m.costo(v)
            if v not in cs or nc < cs[v]: cs[v] = nc; n += 1; heapq.heappush(q, (nc + h(v,metas), n, v)); cf[v] = c
    return [], float('inf'), exp

ALGOS = {"BFS": bfs, "UCS": ucs, "Greedy": greedy, "A*": astar}

# ==================== ESCENARIOS ====================
MAPA_A = """
####################
#S................E#
#.####.####.####...#
#......#.......#...#
#.####.#.#####.#...#
#......#.......#...#
#.####.#.#####.#...#
#......#..S....#...#
#.####.#.#####.#...#
#......#...........#
#..S...#.......#...#
#......#.......#..F#
####################
"""

MAPA_B = """
####################
#S..GGGGGGGGGGGGGGE#
#.##GG####.####....#
#...GGG#.......#...#
#.####.#.#####.#...#
#......#.......#...#
#.####.#.#####.#...#
#......#..S....#...#
#.####.#.#####.#...#
#......#...........#
#..S...#.......#...#
#......#.......#..F#
####################
"""

MAPA_C = """
####################
#S................E#
#.####.####.####...#
#......#.......#...#
#.####.#.#####.#...#
#......#.......#...#
#.####.#.#####.#...#
#......#..S....#...#
#.####.#.#####.#...#
#......#...........#
#..S...#.......#...#
#......#.......#..F#
####################
"""

ESCENARIOS = {
    "A": ("Normal", "Sin incidentes", MAPA_A, []),
    "B": ("Gas", "Gas en ruta directa", MAPA_B, []),
    "C": ("Derrumbe", "Derrumbe en paso 2", MAPA_C, [(2,(1,8)),(2,(1,9)),(2,(1,10))]),
}

# ==================== SIMULACIÓN ====================
def simular(esc_id, algo):
    nombre, desc, mapa, eventos = ESCENARIOS[esc_id]
    m = Mina(mapa)
    starts, metas = m.encontrar('S')[:3], m.encontrar('EF')
    
    # Calcular ruta inicial para cada trabajador
    rutas = {}
    for i, s in enumerate(starts):
        ruta, _, _ = ALGOS[algo](m, s, metas)
        rutas[i] = ruta if ruta else [s]
    
    # Métricas del primer trabajador para la tabla
    path, costo, nodos = ALGOS[algo](m.clonar(), starts[0], metas)
    
    pos = {i: list(starts[i]) for i in range(len(starts))}
    idx = {i: 0 for i in range(len(starts))}
    llegaron = {i: False for i in range(len(starts))}
    pasos = [{"paso": 0, "pos": {f"W{i}": pos[i][:] for i in pos}, "mapa": [''.join(r) for r in m.g], "evento": None}]
    replans = 0
    
    for t in range(1, 50):  # Más pasos para rutas largas
        ev = None
        # Procesar eventos (derrumbes)
        for et, ep in eventos:
            if et == t:
                m.g[ep[0]][ep[1]] = 'X'  # X = derrumbe visible
                ev = f"⚠️ DERRUMBE en ({ep[0]},{ep[1]})"
        
        # Si hubo evento, recalcular rutas para todos
        if ev:
            replans += 1
            for i in range(len(starts)):
                if not llegaron[i]:
                    ruta, _, _ = ALGOS[algo](m, tuple(pos[i]), metas)
                    rutas[i] = ruta if ruta else [tuple(pos[i])]
                    idx[i] = 0
        
        # Mover cada trabajador
        for i in range(len(starts)):
            if llegaron[i]:
                continue
            if rutas[i] and idx[i] < len(rutas[i]) - 1:
                idx[i] += 1
                pos[i] = list(rutas[i][idx[i]])
            # Verificar si llegó a meta
            if tuple(pos[i]) in metas:
                llegaron[i] = True
        
        pasos.append({"paso": t, "pos": {f"W{i}": pos[i][:] for i in pos}, "mapa": [''.join(r) for r in m.g], "evento": ev})
        
        # Terminar si todos llegaron
        if all(llegaron.values()):
            break
    
    return {"esc": f"Escenario {esc_id}", "algo": algo, "path": [list(p) for p in path],
            "costo": costo, "nodos": nodos, "replans": replans, "pasos": pasos}

# ==================== HTML ====================
def generar_html(res):
    data = {f"{r['esc']}_{r['algo']}": r for r in res}
    filas = "\n".join(f'''<tr class="row" data-k="{r['esc']}_{r['algo']}">
        <td>{r['esc']}</td><td><b>{r['algo']}</b></td>
        <td>{len(r['path'])-1}</td><td>{r['costo']:.0f}</td>
        <td>{r['nodos']}</td><td>{r['replans']}</td></tr>''' for r in res)
    
    html = f'''<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>Evacuación Mina - IA</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:Arial;background:linear-gradient(135deg,#1a1a2e,#16213e);color:#fff;padding:20px;min-height:100vh}}
.panel{{background:rgba(255,255,255,0.1);border-radius:12px;padding:20px;margin:15px auto;max-width:1200px}}
h1{{text-align:center;margin-bottom:20px}}
h2{{color:#3498db;border-bottom:2px solid #3498db;padding-bottom:8px;margin-bottom:15px}}
table{{width:100%;border-collapse:collapse}}
th,td{{padding:10px;text-align:center;border-bottom:1px solid rgba(255,255,255,0.1)}}
th{{background:rgba(52,152,219,0.3)}}
.row{{cursor:pointer}}.row:hover,.row.sel{{background:rgba(52,152,219,0.3)}}
#map{{display:grid;gap:2px;justify-content:center;margin:15px 0}}
.cell{{width:28px;height:28px;display:flex;align-items:center;justify-content:center;border-radius:3px;font-size:12px}}
.c-wall{{background:#2c3e50}}.c-floor{{background:#ecf0f1;color:#333}}.c-start{{background:#3498db}}
.c-exit{{background:#27ae60}}.c-refuge{{background:#16a085}}.c-gas{{background:#e74c3c}}.c-collapse{{background:#8b4513}}
.c-path{{background:#f1c40f!important;color:#333}}.c-worker{{background:#e67e22!important;border-radius:50%}}
.controls{{display:flex;gap:10px;justify-content:center;flex-wrap:wrap;margin:15px 0}}
button{{padding:8px 16px;border:none;border-radius:6px;cursor:pointer;background:#3498db;color:#fff}}
button:hover{{background:#2980b9}}
.legend{{display:flex;gap:12px;justify-content:center;flex-wrap:wrap;margin:10px 0}}
.leg{{display:flex;align-items:center;gap:5px}}.leg-c{{width:20px;height:20px;border-radius:3px}}
.info{{background:rgba(0,0,0,0.2);padding:12px;border-radius:8px;margin:10px 0}}
.info-row{{display:flex;justify-content:space-between;padding:4px 0}}
.alert{{background:#e74c3c;padding:10px;border-radius:8px;text-align:center;margin:10px 0;display:none}}
.reas{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:10px 0}}
.reas-item{{background:rgba(0,0,0,0.2);padding:12px;border-radius:8px;text-align:center}}
.reas-item h4{{font-size:1.8em;color:#3498db}}
</style></head><body>
<div class="panel"><h1>⛏️ Evacuación en Mina Subterránea</h1>
<p style="text-align:center;color:#aaa">Comparación: BFS vs UCS vs Greedy vs A*</p></div>

<div class="panel"><h2>🤖 Modelo REAS</h2>
<div class="reas">
<div class="reas-item"><h4>R</h4><b>Reactividad</b><br><small>Reacciona a gas/derrumbes</small></div>
<div class="reas-item"><h4>E</h4><b>Entorno</b><br><small>Mapa dinámico discreto</small></div>
<div class="reas-item"><h4>A</h4><b>Autonomía</b><br><small>Decide ruta sin humanos</small></div>
<div class="reas-item"><h4>S</h4><b>Sociabilidad</b><br><small>Agentes cooperando</small></div>
</div></div>

<div class="panel"><h2>📊 Resultados</h2>
<p style="margin-bottom:10px">Clic en una fila para ver animación</p>
<table><tr><th>Escenario</th><th>Algoritmo</th><th>Pasos</th><th>Costo</th><th>Nodos</th><th>Replans</th></tr>
{filas}</table></div>

<div class="panel"><h2>🎮 Visualización</h2>
<div class="controls">
<button onclick="prev()">⏮ Ant</button><button id="playBtn" onclick="toggle()">▶ Play</button>
<button onclick="next()">Sig ⏭</button><span id="stepTxt">Paso: 0/0</span>
<button onclick="reset()">🔄</button></div>
<div class="legend">
<div class="leg"><div class="leg-c" style="background:#2c3e50"></div>Pared</div>
<div class="leg"><div class="leg-c" style="background:#ecf0f1"></div>Galería</div>
<div class="leg"><div class="leg-c" style="background:#27ae60"></div>Salida</div>
<div class="leg"><div class="leg-c" style="background:#16a085"></div>Refugio</div>
<div class="leg"><div class="leg-c" style="background:#e74c3c"></div>Gas</div>
<div class="leg"><div class="leg-c" style="background:#8b4513"></div>Derrumbe</div>
<div class="leg"><div class="leg-c" style="background:#f1c40f"></div>Ruta</div>
<div class="leg"><div class="leg-c" style="background:#e67e22;border-radius:50%"></div>Trabajador</div>
</div>
<div id="map"></div>
<div id="alert" class="alert"></div>
<div class="info">
<div class="info-row"><span>Escenario:</span><span id="iScen">-</span></div>
<div class="info-row"><span>Algoritmo:</span><span id="iAlgo">-</span></div>
<div class="info-row"><span>Costo:</span><span id="iCost">-</span></div>
<div class="info-row"><span>Nodos:</span><span id="iNodes">-</span></div>
</div></div>

<div class="panel"><h2>📚 Algoritmos</h2>
<div class="reas">
<div class="reas-item"><b>BFS</b><br><small>Menos pasos, ignora costos</small></div>
<div class="reas-item"><b>UCS</b><br><small>Menor costo total</small></div>
<div class="reas-item"><b>Greedy</b><br><small>Rápido, no óptimo</small></div>
<div class="reas-item"><b>A*</b><br><small>Óptimo y eficiente</small></div>
</div></div>

<script>
const DATA=DATA_PLACEHOLDER;
let cur=null,step=0,playing=false,timer=null;

document.querySelectorAll('.row').forEach(r=>r.onclick=()=>select(r));
document.querySelector('.row')?.click();

function select(row){{
  document.querySelectorAll('.row').forEach(r=>r.classList.remove('sel'));
  row.classList.add('sel');
  cur=DATA[row.dataset.k];step=0;stop();
  document.getElementById('iScen').textContent=cur.esc;
  document.getElementById('iAlgo').textContent=cur.algo;
  document.getElementById('iCost').textContent=cur.costo.toFixed(1);
  document.getElementById('iNodes').textContent=cur.nodos;
  render();
}}

function render(){{
  if(!cur)return;
  const s=cur.pasos[step],map=document.getElementById('map'),lines=s.mapa;
  map.style.gridTemplateColumns=`repeat(${{lines[0].length}},28px)`;
  map.innerHTML='';
  for(let r=0;r<lines.length;r++)for(let c=0;c<lines[0].length;c++){{
    const ch=lines[r][c],cell=document.createElement('div');
    cell.className='cell '+({{'#':'c-wall','.':'c-floor','S':'c-start','E':'c-exit','F':'c-refuge','G':'c-gas','X':'c-collapse'}}[ch]||'c-floor');
    if(ch==='G')cell.textContent='☠️';
    if(ch==='E')cell.textContent='🚪';
    if(ch==='F')cell.textContent='🏠';
    if(ch==='X')cell.textContent='💥';
    if(cur.path.some(p=>p[0]===r&&p[1]===c)&&!'EFG'.includes(ch))cell.classList.add('c-path');
    for(const[,pos]of Object.entries(s.pos))if(pos[0]===r&&pos[1]===c){{cell.classList.add('c-worker');cell.textContent='👷';}}
    map.appendChild(cell);
  }}
  document.getElementById('stepTxt').textContent=`Paso: ${{step}}/${{cur.pasos.length-1}}`;
  const alert=document.getElementById('alert');
  if(s.evento){{alert.textContent=s.evento;alert.style.display='block';}}else alert.style.display='none';
}}

function next(){{if(cur&&step<cur.pasos.length-1){{step++;render();}}else stop();}}
function prev(){{if(cur&&step>0){{step--;render();}}}}
function toggle(){{playing?stop():start();}}
function start(){{playing=true;document.getElementById('playBtn').textContent='⏸';timer=setInterval(next,400);}}
function stop(){{playing=false;document.getElementById('playBtn').textContent='▶ Play';clearInterval(timer);}}
function reset(){{step=0;stop();render();}}
</script>
</body></html>'''.replace('DATA_PLACEHOLDER', json.dumps(data, ensure_ascii=False))
    
    with open("evacuation_report.html", "w", encoding="utf-8") as f: f.write(html)
    return os.path.abspath("evacuation_report.html")

# ==================== MAIN ====================
if __name__ == "__main__":
    print("⛏️  EVACUACIÓN MINA - BFS/UCS/Greedy/A*\n")
    res = []
    for e in ["A", "B", "C"]:
        print(f"📍 Escenario {e}: {ESCENARIOS[e][1]}")
        for a in ["BFS", "UCS", "Greedy", "A*"]:
            r = simular(e, a)
            res.append(r)
            print(f"   {a:7} | Pasos:{len(r['path'])-1:3} | Costo:{r['costo']:5.0f} | Nodos:{r['nodos']:4}")
        print()
    
    print(f"✅ Reporte: {generar_html(res)}")
    webbrowser.open('file://' + os.path.abspath("evacuation_report.html"))
