import sys, torch
ref = torch.load(sys.argv[1])["grads"]
for path in sys.argv[2:]:
    g = torch.load(path)["grads"]
    rows = []
    for n in ref:
        if "lora_up" not in n: continue
        a, b = g[n], ref[n]
        rows.append(((a-b).norm().item(), b.norm().item(), ((a-b).norm()/(b.norm()+1e-30)).item(), n))
    tot_err = sum(r[0]**2 for r in rows)**0.5; tot_ref = sum(r[1]**2 for r in rows)**0.5
    print(f"== {path}: 全体 相対L2={tot_err/tot_ref:.3f}  (誤差ノルム上位5モジュール)")
    for r in sorted(rows, reverse=True)[:5]:
        print(f"   err={r[0]:.3e} ref={r[1]:.3e} rel={r[2]:.3f} 寄与={r[0]**2/tot_err**2:.2%} {r[3]}")
    print("   基準ノルム上位3:", [(r[3], f"{r[1]:.2e}") for r in sorted(rows, key=lambda r: -r[1])[:3]])
