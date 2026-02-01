from layout import setup_page
from modules import blackshirt_prob
from modules import modelaccuracy
import footer

df_long, selected_year, selected_group = setup_page("Prediction")

modelaccuracy.render_model_accuracy(selected_year, selected_group)
blackshirt_prob.render_blackshirt_probability(selected_year, selected_group)

footer.footer()






