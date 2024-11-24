from K214142062_BT1_091023 import finta
from finta import TA
import pandas as pd
from pandas import read_excel
from functools import wraps
import numpy as np
from pandas import DataFrame, Series
import mplfinance as mpf
import matplotlib.pyplot as plt


def inputvalidator(input_="ohlc"):
    def dfcheck(func):
        @wraps(func)
        def wrap(*args, **kwargs):

            args = list(args)
            i = 0 if isinstance(args[0], pd.DataFrame) else 1

            args[i] = args[i].rename(columns={c: c.lower() for c in args[i].columns})

            inputs = {
                "o": "open",
                "h": "high",
                "l": "low",
                "c": kwargs.get("column", "close").lower(),
                "v": "volume",
            }

            if inputs["c"] != "close":
                kwargs["column"] = inputs["c"]

            for l in input_:
                if inputs[l] not in args[i].columns:
                    raise LookupError(
                        'Must have a dataframe column named "{0}"'.format(inputs[l])
                    )

            return func(*args, **kwargs)

        return wrap

    return dfcheck


def apply(decorator):
    def decorate(cls):
        for attr in cls.__dict__:
            if callable(getattr(cls, attr)):
                setattr(cls, attr, decorator(getattr(cls, attr)))

        return cls

    return decorate


@apply(inputvalidator(input_="ohlc"))
class TA:

    __version__ = "1.3"

    @classmethod
    def SMA(cls, ohlc: DataFrame, period: int = 41, column: str = "close") -> Series:
        """
        Simple moving average - rolling mean in pandas lingo. Also known as 'MA'.
        The simple moving average (SMA) is the most basic of the moving averages used for trading.
        """

        return pd.Series(
            ohlc[column].rolling(window=period).mean(),
            name="{0} period SMA".format(period),
        )

    @classmethod
    def EMA(
            cls,
            ohlc: DataFrame,
            period: int = 9,
            column: str = "close",
            adjust: bool = True,
    ) -> Series:
        """
        Exponential Weighted Moving Average - Like all moving average indicators, they are much better suited for trending markets.
        When the market is in a strong and sustained uptrend, the EMA indicator line will also show an uptrend and vice-versa for a down trend.
        EMAs are commonly used in conjunction with other indicators to confirm significant market moves and to gauge their validity.
        """


        return pd.Series(
            ohlc[column].ewm(span=period, adjust=adjust).mean(),
            name="{0} period EMA".format(period),
        )

    @classmethod
    def MACD(
            cls,
            ohlc: DataFrame,
            period_fast: int = 12,
            period_slow: int = 26,
            signal: int = 9,
            column: str = "close",
            adjust: bool = True,
    ) -> DataFrame:
        """
        MACD, MACD Signal and MACD difference.
        The MACD Line oscillates above and below the zero line, which is also known as the centerline.
        These crossovers signal that the 12-day EMA has crossed the 26-day EMA. The direction, of course, depends on the direction of the moving average cross.
        Positive MACD indicates that the 12-day EMA is above the 26-day EMA. Positive values increase as the shorter EMA diverges further from the longer EMA.
        This means upside momentum is increasing. Negative MACD values indicates that the 12-day EMA is below the 26-day EMA.
        Negative values increase as the shorter EMA diverges further below the longer EMA. This means downside momentum is increasing.

        Signal line crossovers are the most common MACD signals. The signal line is a 9-day EMA of the MACD Line.
        As a moving average of the indicator, it trails the MACD and makes it easier to spot MACD turns.
        A bullish crossover occurs when the MACD turns up and crosses above the signal line.
        A bearish crossover occurs when the MACD turns down and crosses below the signal line.
        """

        EMA_fast = pd.Series(
            ohlc[column].ewm(ignore_na=False, span=period_fast, adjust=adjust).mean(),
            name="EMA_fast",
        )
        EMA_slow = pd.Series(
            ohlc[column].ewm(ignore_na=False, span=period_slow, adjust=adjust).mean(),
            name="EMA_slow",
        )
        MACD = pd.Series(EMA_fast - EMA_slow, name="MACD")
        MACD_signal = pd.Series(
            MACD.ewm(ignore_na=False, span=signal, adjust=adjust).mean(), name="SIGNAL"
        )

        return pd.concat([MACD, MACD_signal], axis=1)

    @classmethod
    def BBANDS(
            cls,
            ohlc: DataFrame,
            period: int = 20,
            MA: Series = None,
            column: str = "close",
            std_multiplier: float = 2,
    ) -> DataFrame:
        """
         Developed by John Bollinger, Bollinger Bands® are volatility bands placed above and below a moving average.
         Volatility is based on the standard deviation, which changes as volatility increases and decreases.
         The bands automatically widen when volatility increases and narrow when volatility decreases.

         This method allows input of some other form of moving average like EMA or KAMA around which BBAND will be formed.
         Pass desired moving average as <MA> argument. For example BBANDS(MA=TA.KAMA(20)).
         """

        std = ohlc[column].rolling(window=period).std()

        if not isinstance(MA, pd.core.series.Series):
            middle_band = pd.Series(cls.SMA(ohlc, period), name="BB_MIDDLE")
        else:
            middle_band = pd.Series(MA, name="BB_MIDDLE")

        upper_bb = pd.Series(middle_band + (std_multiplier * std), name="BB_UPPER")
        lower_bb = pd.Series(middle_band - (std_multiplier * std), name="BB_LOWER")

        return pd.concat([upper_bb, middle_band, lower_bb], axis=1)

    @classmethod
    def BBWIDTH(
            cls, ohlc: DataFrame, period: int = 20, MA: Series = None, column: str = "close"
    ) -> Series:
        """Bandwidth tells how wide the Bollinger Bands are on a normalized basis."""

        BB = TA.BBANDS(ohlc, period, MA, column)

        return pd.Series(
            (BB["BB_UPPER"] - BB["BB_LOWER"]) / BB["BB_MIDDLE"],
            name="{0} period BBWITH".format(period),
        )

if __name__ == "__main__":
    print([k for k in TA.__dict__.keys() if k[0] not in "_"])

# Đọc dữ liệu từ tệp Excel
file_path = 'VJC20231012.xlsx'  # Thay đổi thành đường dẫn tới tệp Excel của bạn
data = pd.read_excel(file_path)

df = pd.DataFrame(data)
df["date"] = pd.to_datetime(df["date"])
df.set_index("date", inplace=True)

# Calculate the Simple Moving Average (SMA)
sma = finta.TA.SMA(df, period=14, column="close")

# Calculate Bollinger Bands
bbands = finta.TA.BBANDS(df, period=14)

# Plot the stock price data
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["close"], label="Close Price", color="blue")
plt.plot(sma.index, sma, label="SMA", color="orange")
plt.plot(bbands.index, bbands["BB_UPPER"], label="Upper Bollinger Band", color="green")
plt.plot(bbands.index, bbands["BB_LOWER"], label="Lower Bollinger Band", color="red")
plt.fill_between(
    bbands.index, bbands["BB_LOWER"], bbands["BB_UPPER"], color="gray", alpha=0.5
)

plt.title("Stock Price Analysis")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()


# Read the data from Excel and set the date column as the index
data = pd.read_excel(file_path)
data["date"] = pd.to_datetime(data["date"])
data.set_index("date", inplace=True)
# Calculate SMA with a period of 8
sma = finta.TA.SMA(data, period=50, column="close")

# Calculate EMA with a period of 5
ema = finta.TA.EMA(data, period=20, column="close")

# Calculate MA (Simple Moving Average) with a period of 3
ma = finta.TA.SMA(data, period=10, column="close")

# Create the candlestick chart with SMA, EMA, MA, volume, and additional plots
mpf.plot(data.tail(300), type='candle', style='charles',
        title='SPX with SMA, EMA, and MA',
        ylabel='Price (USD)',
        ylabel_lower='Volume',
        volume=True,
        figscale=1.5,
        addplot=[
            mpf.make_addplot(sma, panel=0, color='orange', secondary_y=False, title='SMA'),
            mpf.make_addplot(ema, panel=0, color='blue', secondary_y=False, title='EMA'),
            mpf.make_addplot(ma, panel=0, color='green', secondary_y=False, title='MA')
        ])
mpf.show()

# Calculate MACD
macd = TA.MACD(data, column="close")
macd = macd.tail(300)

# Create an additional plot for MACD
apd_macd = mpf.make_addplot(macd, panel=0, color='purple')  # Secondary Y for MACD

# Create the candlestick chart with MACD, volume, and additional plots
mpf.plot(data.tail(300), type='candle', style='charles',
        title='Stock Data with MACD and Volume',
        ylabel='Price (USD)',
        ylabel_lower='Volume',
        volume=True,
        figscale=1.5,
        addplot=[mpf.make_addplot(macd, panel=1, color='orange', title='MACD')]
        )
mpf.show()
