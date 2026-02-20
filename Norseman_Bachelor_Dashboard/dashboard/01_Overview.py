from layout import setup_page
from modules import participants, timerelations, histograms, rangverlauf_top10, rangverlauf_140_180, rangverlauf_gesamt, whoareweracing, jahres_vergleiche
import footer


df_long, selected_year, selected_group = setup_page("Overview")


participants.render_participants(df_long, selected_year)
timerelations.render_time_relations(df_long)
jahres_vergleiche.render_year_comparison(selected_year, selected_group)
histograms.render(df_long, selected_year, selected_group)
rangverlauf_top10.render_top10_rank_progression()
#rangverlauf_140_180.render_rank_progression_140_180() #vielleicht überflüssig
rangverlauf_gesamt.render_rank_progression_all_athletes()
whoareweracing.render_who_are_we_racing(df_long)


#footer.footer()