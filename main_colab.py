"""
Evacuación en Mina Subterránea - BFS/UCS/Greedy/A*
Taller IA 2026 - Versión Google Colab
"""
from collections import deque
import heapq, json
from IPython.display import HTML, display

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
    
    rutas = {}
    for i, s in enumerate(starts):
        ruta, _, _ = ALGOS[algo](m, s, metas)
        rutas[i] = ruta if ruta else [s]
    
    path, costo, nodos = ALGOS[algo](m.clonar(), starts[0], metas)
    
    pos = {i: list(starts[i]) for i in range(len(starts))}
    idx = {i: 0 for i in range(len(starts))}
    llegaron = {i: False for i in range(len(starts))}
    pasos = [{"paso": 0, "pos": {f"W{i}": pos[i][:] for i in pos}, "mapa": [''.join(r) for r in m.g], "evento": None}]
    replans = 0
    
    for t in range(1, 50):
        ev = None
        for et, ep in eventos:
            if et == t:
                m.g[ep[0]][ep[1]] = 'X'
                ev = f"⚠️ DERRUMBE en ({ep[0]},{ep[1]})"
        
        if ev:
            replans += 1
            for i in range(len(starts)):
                if not llegaron[i]:
                    ruta, _, _ = ALGOS[algo](m, tuple(pos[i]), metas)
                    rutas[i] = ruta if ruta else [tuple(pos[i])]
                    idx[i] = 0
        
        for i in range(len(starts)):
            if llegaron[i]: continue
            if rutas[i] and idx[i] < len(rutas[i]) - 1:
                idx[i] += 1
                pos[i] = list(rutas[i][idx[i]])
            if tuple(pos[i]) in metas: llegaron[i] = True
        
        pasos.append({"paso": t, "pos": {f"W{i}": pos[i][:] for i in pos}, "mapa": [''.join(r) for r in m.g], "evento": ev})
        if all(llegaron.values()): break
    
    return {"esc": f"Escenario {esc_id}", "algo": algo, "path": [list(p) for p in path],
            "costo": costo, "nodos": nodos, "replans": replans, "pasos": pasos}

# ==================== HTML PARA COLAB ====================
def render_mapa(mapa, path, workers_pos):
    """Renderiza un mapa como HTML estatico"""
    path_set = set((p[0], p[1]) for p in path)
    workers_set = set((pos[0], pos[1]) for pos in workers_pos.values())
    
    cells = ""
    for r, row in enumerate(mapa):
        for c, ch in enumerate(row):
            # Determinar clase CSS
            if (r, c) in workers_set:
                cls = "c-worker"
                txt = "W"
            elif ch == '#':
                cls = "c-wall"
                txt = ""
            elif ch == 'G':
                cls = "c-gas"
                txt = "X"
            elif ch == 'E':
                cls = "c-exit"
                txt = "E"
            elif ch == 'F':
                cls = "c-refuge"
                txt = "F"
            elif ch == 'X':
                cls = "c-collapse"
                txt = "!"
            elif (r, c) in path_set:
                cls = "c-path"
                txt = ""
            else:
                cls = "c-floor"
                txt = ""
            cells += f'<div class="cell {cls}">{txt}</div>'
    
    return cells

def generar_html(res):
    filas = "\n".join(f'''<tr style="background:rgba(52,152,219,0.1)">
        <td>{r['esc']}</td><td><b>{r['algo']}</b></td>
        <td>{len(r['path'])-1}</td><td>{r['costo']:.0f}</td>
        <td>{r['nodos']}</td><td>{r['replans']}</td></tr>''' for r in res)
    
    # Generar mapas estaticos para cada resultado
    mapas_html = ""
    for r in res:
        ultimo_paso = r['pasos'][-1]
        mapa_cells = render_mapa(ultimo_paso['mapa'], r['path'], ultimo_paso['pos'])
        mapas_html += f'''
        <div class="mapa-container" style="margin:20px 0;padding:15px;background:rgba(0,0,0,0.2);border-radius:8px">
            <h3 style="color:#3498db;margin-bottom:10px">{r['esc']} - {r['algo']}</h3>
            <p style="color:#aaa;margin-bottom:10px">Pasos: {len(r['path'])-1} | Costo: {r['costo']:.0f} | Nodos: {r['nodos']}</p>
            <div class="grid" style="display:grid;grid-template-columns:repeat(20,24px);gap:1px;justify-content:center">
                {mapa_cells}
            </div>
        </div>'''
    
    html = f'''<div style="font-family:Arial;background:linear-gradient(135deg,#1a1a2e,#16213e);color:#fff;padding:20px;border-radius:12px">
<style>
.cell{{width:24px;height:24px;display:flex;align-items:center;justify-content:center;border-radius:2px;font-size:10px;font-weight:bold}}
.c-wall{{background:#2c3e50}}
.c-floor{{background:#ecf0f1}}
.c-exit{{background:#27ae60;color:#fff}}
.c-refuge{{background:#16a085;color:#fff}}
.c-gas{{background:#e74c3c;color:#fff}}
.c-collapse{{background:#8b4513;color:#fff}}
.c-path{{background:#f1c40f}}
.c-worker{{background:#e67e22;color:#fff;border-radius:50%}}
</style>

<div style="background:rgba(255,255,255,0.1);border-radius:12px;padding:20px;margin:15px 0">
<h1 style="text-align:center;margin-bottom:20px">Evacuacion en Mina Subterranea</h1>
<p style="text-align:center;color:#aaa">Comparacion: BFS vs UCS vs Greedy vs A*</p>
</div>

<div style="background:rgba(255,255,255,0.1);border-radius:12px;padding:20px;margin:15px 0">
<h2 style="color:#3498db;border-bottom:2px solid #3498db;padding-bottom:8px">Modelo REAS</h2>
<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:10px 0">
<div style="background:rgba(0,0,0,0.2);padding:12px;border-radius:8px;text-align:center"><h4 style="font-size:1.8em;color:#3498db">R</h4><b>Reactividad</b><br><small>Reacciona a gas/derrumbes</small></div>
<div style="background:rgba(0,0,0,0.2);padding:12px;border-radius:8px;text-align:center"><h4 style="font-size:1.8em;color:#3498db">E</h4><b>Entorno</b><br><small>Mapa dinamico discreto</small></div>
<div style="background:rgba(0,0,0,0.2);padding:12px;border-radius:8px;text-align:center"><h4 style="font-size:1.8em;color:#3498db">A</h4><b>Autonomia</b><br><small>Decide ruta sin humanos</small></div>
<div style="background:rgba(0,0,0,0.2);padding:12px;border-radius:8px;text-align:center"><h4 style="font-size:1.8em;color:#3498db">S</h4><b>Sociabilidad</b><br><small>Agentes cooperando</small></div>
</div>
</div>

<div style="background:rgba(255,255,255,0.1);border-radius:12px;padding:20px;margin:15px 0">
<h2 style="color:#3498db;border-bottom:2px solid #3498db;padding-bottom:8px">Resultados</h2>
<table style="width:100%;border-collapse:collapse">
<tr style="background:rgba(52,152,219,0.3)"><th style="padding:10px">Escenario</th><th>Algoritmo</th><th>Pasos</th><th>Costo</th><th>Nodos</th><th>Replans</th></tr>
{filas}
</table>
</div>

<div style="background:rgba(255,255,255,0.1);border-radius:12px;padding:20px;margin:15px 0">
<h2 style="color:#3498db;border-bottom:2px solid #3498db;padding-bottom:8px">Leyenda</h2>
<div style="display:flex;gap:15px;flex-wrap:wrap;justify-content:center">
<span><span style="display:inline-block;width:20px;height:20px;background:#2c3e50;border-radius:3px"></span> Pared</span>
<span><span style="display:inline-block;width:20px;height:20px;background:#ecf0f1;border-radius:3px"></span> Galeria</span>
<span><span style="display:inline-block;width:20px;height:20px;background:#27ae60;border-radius:3px"></span> Salida</span>
<span><span style="display:inline-block;width:20px;height:20px;background:#16a085;border-radius:3px"></span> Refugio</span>
<span><span style="display:inline-block;width:20px;height:20px;background:#e74c3c;border-radius:3px"></span> Gas</span>
<span><span style="display:inline-block;width:20px;height:20px;background:#8b4513;border-radius:3px"></span> Derrumbe</span>
<span><span style="display:inline-block;width:20px;height:20px;background:#f1c40f;border-radius:3px"></span> Ruta</span>
<span><span style="display:inline-block;width:20px;height:20px;background:#e67e22;border-radius:50%"></span> Trabajador</span>
</div>
</div>

<div style="background:rgba(255,255,255,0.1);border-radius:12px;padding:20px;margin:15px 0">
<h2 style="color:#3498db;border-bottom:2px solid #3498db;padding-bottom:8px">Mapas Finales</h2>
{mapas_html}
</div>

<div style="background:rgba(255,255,255,0.1);border-radius:12px;padding:20px;margin:15px 0">
<h2 style="color:#3498db;border-bottom:2px solid #3498db;padding-bottom:8px">Algoritmos</h2>
<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px">
<div style="background:rgba(0,0,0,0.2);padding:12px;border-radius:8px;text-align:center"><b>BFS</b><br><small>Menos pasos, ignora costos</small></div>
<div style="background:rgba(0,0,0,0.2);padding:12px;border-radius:8px;text-align:center"><b>UCS</b><br><small>Menor costo total</small></div>
<div style="background:rgba(0,0,0,0.2);padding:12px;border-radius:8px;text-align:center"><b>Greedy</b><br><small>Rapido, no optimo</small></div>
<div style="background:rgba(0,0,0,0.2);padding:12px;border-radius:8px;text-align:center"><b>A*</b><br><small>Optimo y eficiente</small></div>
</div>
</div>

</div>'''
    
    return html

# ==================== EJECUTAR ====================
def run():
    print("⛏️  EVACUACIÓN MINA - BFS/UCS/Greedy/A*\n")
    res = []
    for e in ["A", "B", "C"]:
        print(f"📍 Escenario {e}: {ESCENARIOS[e][1]}")
        for a in ["BFS", "UCS", "Greedy", "A*"]:
            r = simular(e, a)
            res.append(r)
            print(f"   {a:7} | Pasos:{len(r['path'])-1:3} | Costo:{r['costo']:5.0f} | Nodos:{r['nodos']:4}")
        print()
    
    display(HTML(generar_html(res)))

# Ejecutar automáticamente
run()
