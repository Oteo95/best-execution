import matplotlib.pyplot as plt


def plot_results(env, mode="max"):
    my_df, market_df, mpx = env.stats_df()
    my_vwap = my_df["vwap"]
    my_vol = my_df["vol"]
    vwap = market_df["vwap"]
    mkt_vol = market_df["vol"]
    norm_my_vol = (my_vol > 0) * 0.1
    if mode == "max":
        norm_mkt_vol = mkt_vol / mkt_vol.max()
    else:
        norm_mkt_vol = mkt_vol / mkt_vol.quantile(q=0.99)
    norm_mkt_vol = norm_mkt_vol.where(norm_mkt_vol < 1, 1)
    cum_mkt_vol = mkt_vol.cumsum() / mkt_vol.sum()
    cum_my_vol = my_vol.cumsum() / env.vol_care
    plt.figure(1, figsize=(15, 10))
    plt.subplot(211)
    plt.title("Market VS Algo")
    plt.plot(vwap, color="blue")
    plt.plot(my_vwap, color="orange")
    plt.plot(mpx, color="red")
    plt.legend(['vwap', 'algo', 'mid_price'])
    plt.grid(True)
    plt.subplot(212)
    plt.plot(cum_mkt_vol, color="blue")
    plt.plot(cum_my_vol, color="orange")
    plt.bar(
        mkt_vol.index,
        norm_mkt_vol,
        width=0.00005,
        color="blue",
        alpha=0.6
    )
    plt.bar(
        my_vol.index,
        norm_my_vol,
        width=0.00007,
        color="orange",
        alpha=0.6
    )
    plt.legend(["algo cumvol", "mkt cumvol", "algo vol ", "mkt vol"])
    plt.grid(True)
