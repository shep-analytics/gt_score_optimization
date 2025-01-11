import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

def make_output(actions, training_results=None, testing_results=None, best_params=None,
                portfolio_values_training=None, portfolio_values_testing=None):
    """
    Creates a single PDF file (output.pdf) containing all the requested output.
    
    Parameters:
    - actions (list): list of strings specifying which outputs to include
    - training_results, testing_results (dict): dictionaries of performance metrics, etc.
    - best_params (dict): best hyperparameters found
    - portfolio_values_training, portfolio_values_testing (pd.DataFrame): dataframes with
      'date_time' and 'value' columns to plot
    """

    with PdfPages("output.pdf") as pdf:

        if "print_best_training_data_results" in actions and training_results is not None:
            fig, ax = plt.subplots()
            ax.axis("off")
            text_str = (
                f"Best Training Results:\n"
                f"Best loss: {training_results.get('best_loss', 'N/A')}\n"
                f"Best params: {best_params}\n"
                f"Best strategy: {training_results.get('best_strategy', 'N/A')}"
            )
            ax.text(0.1, 0.7, text_str, fontsize=12, wrap=True)
            pdf.savefig(fig)
            plt.close(fig)

        if "print_tested_data_results" in actions and testing_results is not None:
            fig, ax = plt.subplots()
            ax.axis("off")
            text_str = (
                f"Testing (Backtest) Results:\n"
                f"Total Money Made: ${round(testing_results.get('total_amount_of_money_made', 0), 2)}\n"
                f"Total Return: {round(testing_results.get('total_percentage_gain', 0)*100, 2)}%\n"
                f"Number of Trades: {testing_results.get('total_trades', 'N/A')}\n"
                f"Average Hold Time: {testing_results.get('average_time_holding_position', 'N/A')}\n"
                f"Average Return per Year: {round(testing_results.get('average_return_per_year', 0)*100, 2)}%\n"
                f"Average Trades per Year: {testing_results.get('average_trades_per_year', 'N/A')}"
            )
            ax.text(0.1, 0.7, text_str, fontsize=12, wrap=True)
            pdf.savefig(fig)
            plt.close(fig)

        if "plot_portfolio_values_from_training_data_results" in actions and portfolio_values_training is not None:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(portfolio_values_training["date_time"], portfolio_values_training["value"])
            ax.set_title("Training Portfolio Value Over Time")
            ax.set_xlabel("Time")
            ax.set_ylabel("Portfolio Value")
            pdf.savefig(fig)
            plt.close(fig)

        if "plot_portfolio_values_from_testing_data_results" in actions and portfolio_values_testing is not None:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(portfolio_values_testing["date_time"], portfolio_values_testing["value"])
            ax.set_title("Testing Portfolio Value Over Time")
            ax.set_xlabel("Time")
            ax.set_ylabel("Portfolio Value")
            pdf.savefig(fig)
            plt.close(fig)

    print("All output saved to output.pdf.")
