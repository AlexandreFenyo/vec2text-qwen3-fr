Les figures sont dessinées en SVG (source) et exportées en PNG pour l'affichage dans le markdown :
    .venv/bin/python -c "import cairosvg; [cairosvg.svg2png(url=f'docs/img/{n}.svg', write_to=f'docs/img/{n}.png', output_width=1600) for n in ('encodage_decodage','rotation_plan_3d')]"
