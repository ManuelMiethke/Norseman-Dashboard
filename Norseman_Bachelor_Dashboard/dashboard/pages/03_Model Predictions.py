from layout import setup_page
from modules import blackshirt_prob
from modules import modelaccuracy
from modules import frontier_pace
from modules import probability_3d
import footer

df_long, selected_year, selected_group = setup_page("Prediction")

modelaccuracy.render_model_accuracy(selected_year, selected_group)
blackshirt_prob.render_blackshirt_probability(selected_year, selected_group)
frontier_pace.render_frontier_pace(selected_year, selected_group)
probability_3d.render_probability_3d(selected_year, selected_group)

footer.footer()




