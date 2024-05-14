
#연간 수익률 8% 이상, 배당금 3% 이상
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd


start_date = '2011 -12-1'  # 1 December 2020
end_date = '2023-2-2'     # 2 February 2023
# "start_date" must be an older date than the "end_date"

amazon = pd.DataFrame(yf.download(tickers="AMZN",
                     start=start_date,
                     end=end_date)['Open'])


if __name__ == '__main__':
    print(amazon)
    plt.plot(amazon)
    plt.title('amazon')

    plt.show()