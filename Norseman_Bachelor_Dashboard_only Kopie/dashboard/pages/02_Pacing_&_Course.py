from layout import setup_page
from modules import accumulated
from modules import weatherspeed
from modules import pacetabelle
from modules import heatmap
import footer

df_long, selected_year, selected_group = setup_page("Pacing & Elevation")

weatherspeed.render_weatherspeed(selected_year, selected_group)
accumulated.render(df_long, selected_year, selected_group)
pacetabelle.render_pace_boxplots_with_tables(selected_year, selected_group)
heatmap.render_pacing_heatmap(selected_year, selected_group)
footer.footer()
