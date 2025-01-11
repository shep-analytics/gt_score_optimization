import matplotlib.pyplot as plt
import pandas as pd
import os
from fpdf import FPDF
from machine_learning import optimize
from machine_learning import loss_functions
from data_collection.fetch_data import fetch_historical_data
from strategies.import_all import *
import modules.backtester as backtest

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Strategy Analysis Report', 0, 1, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()

    def add_image(self, image_path):
        self.image(image_path, x=10, y=None, w=190)

tickers = ['SPY']

training_data = {"start": '1980-01-01', "end": '2008-01-01'}
testing_data = {"start": '2008-01-01', "end": '2009-01-01'}
loss_functions_list = [
    loss_functions.simple_loss_function,
    loss_functions.sharpe_ratio_loss_function,
    loss_functions.ridge_regression_loss_function,
    loss_functions.elastic_net_loss_function,
    loss_functions.gt_function
]
optimization_techniques = ["random", "hyperopt", "genetic"]
max_evals = 100
pop_size = 10

data_frames = [fetch_historical_data(ticker, '1d', training_data['start'], training_data['end']) for ticker in tickers]
test_data_frames = [fetch_historical_data(ticker, '1d', testing_data['start'], testing_data['end']) for ticker in tickers]

pdf = PDFReport()
pdf.add_page()

all_images = []

# Add report title with tickers, dates, max evals, and pop size
report_title = (
    f"Tickers: {', '.join(tickers)}\n"
    f"Training Data: {training_data['start']} to {training_data['end']}\n"
    f"Testing Data: {testing_data['start']} to {testing_data['end']}\n"
    f"Max Evaluations: {max_evals}\n"
    f"Population Size: {pop_size}\n"
)
pdf.chapter_title("Report Overview")
pdf.chapter_body(report_title)

def run_loss_function(opt_tech, loss_function,pdf):
    pdf.chapter_title(f"Loss Function: {loss_function.__name__}, Optimization Technique: {opt_tech}")
    
    # Optimization and Backtesting
    results = optimize.optimize(strategies, data_frames, loss_function, opt_tech, max_evals=max_evals, population_size=pop_size)
    combined_results_training = optimize.compile_backtest_results_sequential([
        backtest.run_backtest(results['best_strategy'](df['ohlc'], results['best_params']))[0] for df in data_frames
    ], data_frames)
    combined_results_testing = optimize.compile_backtest_results_sequential([
        backtest.run_backtest(results['best_strategy'](df['ohlc'], results['best_params']))[0] for df in test_data_frames
    ], test_data_frames)

    # Add textual results
    body = (
        f"Training Data Results:\n"
        f"- Best Loss Achieved: {round(results['best_loss'], 4)}\n\n"  # Add the loss value here
        f"- Total Money Made: ${round(combined_results_training['total_amount_of_money_made'], 2)}\n"
        f"- Total Return: {round(combined_results_training['total_percentage_gain'] * 100, 2)}%\n"
        f"- Market Return: {round(((combined_results_training['portfolio_values_over_time'][-1]['stock_value'] - combined_results_training['portfolio_values_over_time'][0]['stock_value']) / combined_results_training['portfolio_values_over_time'][0]['stock_value']) * 100, 2)}%\n"
        f"- Number of Trades: {combined_results_training['total_trades']}\n"
        f"- Average Hold Time: {combined_results_training['average_time_holding_position']}\n"
        f"- Average Return per Year: {round(combined_results_training['average_return_per_year'] * 100, 2)}%\n"
        f"- Average Trades per Year: {combined_results_training['average_trades_per_year']}\n\n"
        f"Validation Data Results:\n"
        f"- Total Money Made: ${round(combined_results_testing['total_amount_of_money_made'], 2)}\n"
        f"- Total Return: {round(combined_results_testing['total_percentage_gain'] * 100, 2)}%\n"
        f"- Market Return: {round(((combined_results_testing['portfolio_values_over_time'][-1]['stock_value'] - combined_results_testing['portfolio_values_over_time'][0]['stock_value']) / combined_results_testing['portfolio_values_over_time'][0]['stock_value']) * 100, 2)}%\n"
        f"- Number of Trades: {combined_results_testing['total_trades']}\n"
        f"- Average Hold Time: {combined_results_testing['average_time_holding_position']}\n"
        f"- Average Return per Year: {round(combined_results_testing['average_return_per_year'] * 100, 2)}%\n"
        f"- Average Trades per Year: {combined_results_testing['average_trades_per_year']}\n"
    )
    pdf.chapter_body(body)

    # Create training and testing graphs
    for phase, combined_results in [('Training', combined_results_training), ('Testing', combined_results_testing)]:
        portfolio_values = pd.DataFrame(combined_results['portfolio_values_over_time'])
        portfolio_values['time_elapsed'] = range(len(portfolio_values))
        plt.figure(figsize=(10, 6))
        plt.plot(portfolio_values['time_elapsed'], portfolio_values['value'], label=f'{phase} Portfolio Value')
        plt.xlabel('Time (Months)')
        plt.ylabel('Portfolio Value')
        plt.title(f'{phase} - {loss_function.__name__} ({opt_tech})')
        plt.legend()
        plt.grid()
        image_path = f'{loss_function.__name__}_{opt_tech}_{phase.lower()}.png'
        plt.savefig(image_path, dpi=300)
        plt.close()
        all_images.append(image_path)
        pdf.add_image(image_path)
    
    # Combined graph for both training and testing
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(combined_results_training['portfolio_values_over_time'])), 
                pd.DataFrame(combined_results_training['portfolio_values_over_time'])['value'], label='Training')
    plt.plot(range(len(combined_results_testing['portfolio_values_over_time'])), 
                pd.DataFrame(combined_results_testing['portfolio_values_over_time'])['value'], label='Testing')
    plt.xlabel('Time (Trading Days)')
    plt.ylabel('Portfolio Value')
    plt.title(f'Combined Portfolio Value - {loss_function.__name__} ({opt_tech})')
    plt.legend()
    plt.grid()
    combined_image_path = f'{loss_function.__name__}_{opt_tech}_combined.png'
    plt.savefig(combined_image_path, dpi=300)
    plt.close()
    all_images.append(combined_image_path)
    pdf.add_image(combined_image_path)

    # Aggregate graph for all loss functions
    for phase, data_frames_set in [('Training', data_frames), ('Testing', test_data_frames)]:
        plt.figure(figsize=(10, 6))
        for loss_function in loss_functions_list:
            results = optimize.optimize(strategies, data_frames_set, loss_function, opt_tech, max_evals=max_evals, population_size=pop_size)
            combined_results = optimize.compile_backtest_results_sequential([
                backtest.run_backtest(results['best_strategy'](df['ohlc'], results['best_params']))[0] for df in data_frames_set
            ], data_frames_set)
            portfolio_values = pd.DataFrame(combined_results['portfolio_values_over_time'])
            plt.plot(range(len(portfolio_values)), portfolio_values['value'], label=loss_function.__name__)
        plt.xlabel('Time (Months)')
        plt.ylabel('Portfolio Value')
        plt.title(f'{phase} Portfolio Comparison - {opt_tech}')
        plt.legend()
        plt.grid()
        comparison_image_path = f'{opt_tech}_{phase.lower()}_comparison.png'
        plt.savefig(comparison_image_path, dpi=300)
        plt.close()
        all_images.append(comparison_image_path)
        pdf.add_image(comparison_image_path)

for opt_tech in optimization_techniques:
    for loss_function in loss_functions_list:
        run_loss_function(opt_tech,loss_function,pdf)


# Save the PDF
pdf.output("strategy_analysis_report.pdf")
print("PDF report saved as 'strategy_analysis_report.pdf'.")

# Clean up images
for image in all_images:
    if os.path.exists(image):
        os.remove(image)
print("Temporary images deleted.")
