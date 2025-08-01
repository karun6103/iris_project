//+------------------------------------------------------------------+
//| MM Flip CodePro Advanced EA                                      |
//+------------------------------------------------------------------+
#property strict

//--- Input parameters
input double   SAR_Period         = 0.56;
input double   SAR_Step           = 0.25;
input double   SAR_Acceleration   = 0.09;
input int      LookbackCandles    = 50;
input double   LotSize            = 0.05;
input int      TrailingStopPips   = 25;
input int      StopLossPips       = 530;
input int      MaxSpreadPoints    = 20;
input int      MagicNumber        = 222222;
input bool     NewsFilter         = true;
input int      NewsPauseBefore    = 30; // minutes
input int      NewsPauseAfter     = 30; // minutes
input string   NewsCurrencies     = "USD,EUR,CAD,AUD,NZD,GBP";

//--- Chart annotation settings
input color    TradeLabelColor    = clrRed;
input color    TradeLineColor     = clrRed;
input int      TradeLabelFontSize = 8;

//--- Global variables
datetime lastTradeTime = 0;

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    // 1. News Filter
    if(NewsFilter && IsNewsTime()) return;

    // 2. Spread Check
    if(MarketInfo(Symbol(), MODE_SPREAD) > MaxSpreadPoints) return;

    // 3. Trend Detection
    int trend = GetTrendDirection();
    if(trend == 0) return; // No clear trend

    // 4. Entry Logic
    if(CanOpenTrade(trend))
    {
        OpenTrade(trend);
    }

    // 5. Trade Management
    ManageOpenTrades();
}

//+------------------------------------------------------------------+
//| Trend Detection using Parabolic SAR and Lookback                 |
//+------------------------------------------------------------------+
int GetTrendDirection()
{
    double sarPrev = iSAR(NULL, 0, SAR_Period, SAR_Step, 1);
    double sarCurr = iSAR(NULL, 0, SAR_Period, SAR_Step, 0);
    double pricePrev = iClose(NULL, 0, 1);
    double priceCurr = iClose(NULL, 0, 0);

    // Simple SAR-based trend logic
    if(priceCurr > sarCurr && pricePrev > sarPrev)
        return 1; // Uptrend
    if(priceCurr < sarCurr && pricePrev < sarPrev)
        return -1; // Downtrend

    return 0; // No clear trend
}

//+------------------------------------------------------------------+
//| Entry Filter: Lookback Pattern                                   |
//+------------------------------------------------------------------+
bool CanOpenTrade(int trend)
{
    // Check for existing trades
    if(CountOpenTrades() > 0) return false;

    // Lookback pattern: e.g., avoid if last 50 candles are choppy
    int up = 0, down = 0;
    for(int i=1; i<=LookbackCandles; i++)
    {
        if(iClose(NULL,0,i) > iOpen(NULL,0,i)) up++;
        else if(iClose(NULL,0,i) < iOpen(NULL,0,i)) down++;
    }
    if(MathMax(up,down) < LookbackCandles*0.6) return false; // Not trending enough

    return true;
}

//+------------------------------------------------------------------+
//| Open Trade and Annotate Chart                                    |
//+------------------------------------------------------------------+
void OpenTrade(int trend)
{
    double sl = 0, tp = 0;
    double price = (trend == 1) ? Ask : Bid;
    int slippage = 3;

    if(trend == 1)
        sl = price - StopLossPips * Point;
    else
        sl = price + StopLossPips * Point;

    int ticket = OrderSend(Symbol(), trend==1?OP_BUY:OP_SELL, LotSize, price, slippage, sl, tp, "MM Flip CodePro", MagicNumber, 0, clrBlue);
    if(ticket > 0)
    {
        lastTradeTime = TimeCurrent();
        AnnotateTrade(trend, price, ticket);
    }
}

//+------------------------------------------------------------------+
//| Manage Open Trades: Trailing Stop                                |
//+------------------------------------------------------------------+
void ManageOpenTrades()
{
    for(int i=0; i<OrdersTotal(); i++)
    {
        if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
        {
            if(OrderMagicNumber() != MagicNumber || OrderSymbol() != Symbol()) continue;

            double newStop = 0;
            if(OrderType() == OP_BUY)
            {
                newStop = Bid - TrailingStopPips * Point;
                if(OrderStopLoss() < newStop)
                    OrderModify(OrderTicket(), OrderOpenPrice(), newStop, 0, 0, clrBlue);
            }
            else if(OrderType() == OP_SELL)
            {
                newStop = Ask + TrailingStopPips * Point;
                if(OrderStopLoss() > newStop || OrderStopLoss() == 0)
                    OrderModify(OrderTicket(), OrderOpenPrice(), newStop, 0, 0, clrRed);
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Count Open Trades for this EA                                    |
//+------------------------------------------------------------------+
int CountOpenTrades()
{
    int count = 0;
    for(int i=0; i<OrdersTotal(); i++)
    {
        if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
        {
            if(OrderMagicNumber() == MagicNumber && OrderSymbol() == Symbol())
                count++;
        }
    }
    return count;
}

//+------------------------------------------------------------------+
//| News Filter (Simulated)                                          |
//+------------------------------------------------------------------+
bool IsNewsTime()
{
    // Placeholder: In real use, integrate with a news indicator or file
    // For now, always return false (no news)
    // You can add logic to read from a CSV or global variable set by a news indicator
    return false;
}

//+------------------------------------------------------------------+
//| Annotate Trade on Chart                                          |
//+------------------------------------------------------------------+
void AnnotateTrade(int trend, double price, int ticket)
{
    string label = (trend == 1) ? "BUY " : "SELL ";
    label += DoubleToStr(LotSize, 2) + " #" + IntegerToString(ticket);

    // Draw label at the trade price on the current bar
    string objName = "TradeLabel_" + IntegerToString(ticket);
    ObjectCreate(0, objName, OBJ_TEXT, 0, Time[0], price);
    ObjectSetText(objName, label, TradeLabelFontSize, "Arial", TradeLabelColor);

    // Draw a horizontal line at the trade price
    string lineName = "TradeLine_" + IntegerToString(ticket);
    ObjectCreate(0, lineName, OBJ_HLINE, 0, 0, price);
    ObjectSetInteger(0, lineName, OBJPROP_COLOR, TradeLineColor);
    ObjectSetInteger(0, lineName, OBJPROP_WIDTH, 1);
    ObjectSetInteger(0, lineName, OBJPROP_STYLE, STYLE_DOT);
}

//+------------------------------------------------------------------+
//| Deinit: Clean up chart objects                                   |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    // Remove all trade labels and lines
    for(int i=ObjectsTotal()-1; i>=0; i--)
    {
        string name = ObjectName(i);
        if(StringFind(name, "TradeLabel_") == 0 || StringFind(name, "TradeLine_") == 0)
            ObjectDelete(name);
    }
}

//+------------------------------------------------------------------+