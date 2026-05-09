"""
make_combined_table.py
======================
Merges fig3 CSV results into a single combined LaTeX table.

Format:
    Rows    = Q values
    Columns = (PSD, λ-SPSD) × (Null, Perturbed, Student, Laplace)

Usage:
    python scripts/make_combined_table.py --save_latex results/table_fig3_combined.tex
"""

import argparse
import csv

EXPERIMENTS = [
    ("null",            "StandardGaussian",  "results/fig3_null.csv"),
    ("perturbed_gauss", "PerturbedGaussian", "results/fig3_perturbed.csv"),
    ("gauss_t",         "Student-$t$",       "results/fig3_student.csv"),
    ("laplace",         "Laplace",           "results/fig3_laplace.csv"),
]

def load_csv(path):
    """Returns {Q: {psd_rej, wpsd_rej, psd_msnr, wpsd_msnr}}"""
    data = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            Q = int(row['Q'])
            data[Q] = {
                'psd_rej':   float(row['psd_rej_mean']),
                'wpsd_rej':  float(row['wpsd_rej_mean']),
                'psd_msnr':  float(row['psd_msnr_mean']),
                'wpsd_msnr': float(row['wpsd_msnr_mean']),
            }
    return data

def _bold(s):
    return r'\textbf{' + s + '}'

def make_table(metric_key, caption, fmt='{:.3f}'):
    """Build one LaTeX table for a given metric (rejection rate or MSNR)."""
    
    all_data = {name: load_csv(path) for _, name, path in EXPERIMENTS}
    Q_list   = sorted(next(iter(all_data.values())).keys())
    titles   = [title for _, title, _ in EXPERIMENTS]
    ncols    = len(EXPERIMENTS)

    col_spec = 'r' + ' rr' * ncols
    lines = [
        r'\begin{table}[ht]',
        r'\centering',
        r'\caption{' + caption + r'}',
        r'\resizebox{\textwidth}{!}{',
        r'\begin{tabular}{' + col_spec + r'}',
        r'\toprule',
    ]

    # Header row 1 — experiment names
    header1 = r'$Q$'
    for title in titles:
        header1 += r' & \multicolumn{2}{c}{' + title + r'}'
    lines.append(header1 + r' \\')

    # Cmidrules under each experiment pair
    cmidrules = ''
    for i in range(ncols):
        s = 2 + 2 * i
        cmidrules += r'\cmidrule(lr){' + f'{s}-{s+1}' + '}'
    lines.append(cmidrules)

    # Header row 2 — method names
    header2 = ''
    for _ in titles:
        header2 += r' & $\mathrm{PSD}_{L_1}$ & $\lambda$-$\mathrm{SPSD}_{L_1}$'
    lines.append(header2 + r' \\')
    lines.append(r'\midrule')

    # Data rows
    psd_key  = 'psd_'  + metric_key
    wpsd_key = 'wpsd_' + metric_key

    for Q in Q_list:
        row = str(Q)
        for _, title, _ in EXPERIMENTS:
            d    = all_data[title][Q]
            pv   = d[psd_key]
            wv   = d[wpsd_key]
            ps   = fmt.format(pv)
            ws   = fmt.format(wv)
            if   wv > pv: ws = _bold(ws)
            elif pv > wv: ps = _bold(ps)
            row += f' & {ps} & {ws}'
        lines.append(row + r' \\')

    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'}',   # end resizebox
        r'\end{table}',
    ]
    return '\n'.join(lines) + '\n'


def main(save_latex):
    rej_table  = make_table('rej',  
                            r'Rejection rate: $\mathrm{PSD}_{L_1}$ vs $\lambda$-$\mathrm{SPSD}_{L_1}$ across distributions ($d=16$, $\alpha=0.05$)',
                            fmt='{:.3f}')
    msnr_table = make_table('msnr', 
                            r'MSNR: $\mathrm{PSD}_{L_1}$ vs $\lambda$-$\mathrm{SPSD}_{L_1}$ across distributions ($d=16$)',
                            fmt='{:.2f}')

    combined = rej_table + '\n' + msnr_table

    if save_latex:
        with open(save_latex, 'w') as f:
            f.write(combined)
        print(f"Saved → {save_latex}")
    else:
        print(combined)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--save_latex", type=str, default=None)
    args = p.parse_args()
    main(args.save_latex)