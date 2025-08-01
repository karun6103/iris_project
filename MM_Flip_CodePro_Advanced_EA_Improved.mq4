//+------------------------------------------------------------------+
//| MM Flip CodePro Advanced EA - Improved Version                  |
//+------------------------------------------------------------------+
#property strict

//--- Input parameters
input double   SAR_Period         = 0.02;    // Fixed: SAR period should be step, not >1
input double   SAR_Step           = 0.02;    // Fixed: Typical SAR step value
input double   SAR_Acceleration   = 0.2;     // Fixed: Typical SAR acceleration
input int      LookbackCandles    = 50;
input double   LotSize            = 0.01;    // Reduced for safer trading
input int      TrailingStopPips   = 25;
input int      StopLossPips       = 100;     // Reduced from 530 (too large)
input int      TakeProfitPips     = 50;      // Added take profit
input int      MaxSpreadPoints    = 20;
input int      MagicNumber        = 222222;
input bool     NewsFilter         = true;
input int      NewsPauseBefore    = 30;      // minutes
input int      NewsPauseAfter     = 30;      // minutes
input string   NewsCurrencies     = "USD,EUR,CAD,AUD,NZD,GBP";
input int      MinTimeBetweenTrades = 60;    // minutes between trades

//--- Chart annotation settings
input color    TradeLabelColor    = clrRed;
input color    TradeLineColor     = clrRed;
input int      TradeLabelFontSize = 8;

//--- Global variables
datetime lastTradeTime = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    // Validate input parameters
    if(LotSize <= 0 || StopLossPips <= 0 || TrailingStopPips <= 0)
    {
        Print("Invalid input parameters!");
        return(INIT_PARAMETERS_INCORRECT);
    }
    
    Print("MM Flip CodePro Advanced EA initialized successfully");
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    // 1. News Filter
    if(NewsFilter && IsNewsTime()) return;

    // 2. Spread Check
    double spread = MarketInfo(Symbol(), MODE_SPREAD);
    if(spread > MaxSpreadPoints) 
    {
        Print("Spread too high: ", spread, " points");
        return;
    }

    // 3. Time filter - minimum time between trades
    if(TimeCurrent() - lastTradeTime < MinTimeBetweenTrades * 60) return;

    // 4. Trend Detection
    int trend = GetTrendDirection();
    if(trend == 0) return; // No clear trend

    // 5. Entry Logic
    if(CanOpenTrade(trend))
    {
        OpenTrade(trend);
    }

    // 6. Trade Management
    ManageOpenTrades();
}

//+------------------------------------------------------------------+
//| Trend Detection using Parabolic SAR and Lookback                 |
//+------------------------------------------------------------------+
int GetTrendDirection()
{
    // Check if we have enough bars
    if(Bars < LookbackCandles + 5) return 0;
    
    double sarPrev = iSAR(NULL, 0, SAR_Step, SAR_Acceleration, 1);
    double sarCurr = iSAR(NULL, 0, SAR_Step, SAR_Acceleration, 0);
    double pricePrev = iClose(NULL, 0, 1);
    double priceCurr = iClose(NULL, 0, 0);

    // Enhanced SAR-based trend logic with confirmation
    bool upTrend = (priceCurr > sarCurr && pricePrev > sarPrev);
    bool downTrend = (priceCurr < sarCurr && pricePrev < sarPrev);
    
    // Additional confirmation: check if SAR is moving in trend direction
    if(upTrend && sarCurr < sarPrev) return 1;   // Uptrend confirmed
    if(downTrend && sarCurr > sarPrev) return -1; // Downtrend confirmed

    return 0; // No clear trend
}

//+------------------------------------------------------------------+
//| Entry Filter: Lookback Pattern                                   |
//+------------------------------------------------------------------+
bool CanOpenTrade(int trend)
{
    // Check for existing trades
    if(CountOpenTrades() > 0) return false;

    // Enhanced lookback pattern analysis
    int up = 0, down = 0, doji = 0;
    double avgRange = 0;
    
    for(int i=1; i<=LookbackCandles; i++)
    {
        double open = iOpen(NULL, 0, i);
        double close = iClose(NULL, 0, i);
        double high = iHigh(NULL, 0, i);
        double low = iLow(NULL, 0, i);
        
        avgRange += high - low;
        
        if(MathAbs(close - open) < (high - low) * 0.1) // Doji-like candle
            doji++;
        else if(close > open) 
            up++;
        else if(close < open) 
            down++;
    }
    
    avgRange /= LookbackCandles;
    double currentRange = iHigh(NULL, 0, 0) - iLow(NULL, 0, 0);
    
    // Avoid trading in low volatility or choppy conditions
    if(doji > LookbackCandles * 0.3) return false; // Too many doji candles
    if(currentRange < avgRange * 0.5) return false; // Current bar too small
    if(MathMax(up, down) < LookbackCandles * 0.55) return false; // Not trending enough

    // Trend alignment check
    if(trend == 1 && up <= down) return false;
    if(trend == -1 && down <= up) return false;

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
    
    // Calculate stop loss and take profit
    double pipValue = Point;
    if(Digits == 5 || Digits == 3) pipValue = Point * 10; // Account for 5-digit brokers
    
    if(trend == 1) // Buy order
    {
        sl = price - StopLossPips * pipValue;
        tp = price + TakeProfitPips * pipValue;
    }
    else // Sell order
    {
        sl = price + StopLossPips * pipValue;
        tp = price - TakeProfitPips * pipValue;
    }

    // Validate stop loss and take profit levels
    double minStopLevel = MarketInfo(Symbol(), MODE_STOPLEVEL) * Point;
    if(trend == 1)
    {
        if(sl > price - minStopLevel) sl = price - minStopLevel;
        if(tp < price + minStopLevel) tp = price + minStopLevel;
    }
    else
    {
        if(sl < price + minStopLevel) sl = price + minStopLevel;
        if(tp > price - minStopLevel) tp = price - minStopLevel;
    }

    int ticket = OrderSend(Symbol(), trend==1?OP_BUY:OP_SELL, LotSize, price, slippage, sl, tp, "MM Flip CodePro", MagicNumber, 0, clrBlue);
    
    if(ticket > 0)
    {
        lastTradeTime = TimeCurrent();
        AnnotateTrade(trend, price, ticket);
        Print("Trade opened: ", trend==1?"BUY":"SELL", " Ticket: ", ticket, " Price: ", price);
    }
    else
    {
        Print("Failed to open trade. Error: ", GetLastError());
    }
}

//+------------------------------------------------------------------+
//| Manage Open Trades: Trailing Stop                                |
//+------------------------------------------------------------------+
void ManageOpenTrades()
{
    double pipValue = Point;
    if(Digits == 5 || Digits == 3) pipValue = Point * 10;
    
    for(int i=0; i<OrdersTotal(); i++)
    {
        if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
        {
            if(OrderMagicNumber() != MagicNumber || OrderSymbol() != Symbol()) continue;

            double newStop = 0;
            double minStopLevel = MarketInfo(Symbol(), MODE_STOPLEVEL) * Point;
            
            if(OrderType() == OP_BUY)
            {
                newStop = Bid - TrailingStopPips * pipValue;
                // Ensure minimum distance and only move stop loss up
                if(newStop > OrderStopLoss() + pipValue && newStop < Bid - minStopLevel)
                {
                    if(!OrderModify(OrderTicket(), OrderOpenPrice(), newStop, OrderTakeProfit(), 0, clrBlue))
                        Print("Failed to modify BUY order. Error: ", GetLastError());
                }
            }
            else if(OrderType() == OP_SELL)
            {
                newStop = Ask + TrailingStopPips * pipValue;
                // Ensure minimum distance and only move stop loss down
                if((OrderStopLoss() == 0 || newStop < OrderStopLoss() - pipValue) && newStop > Ask + minStopLevel)
                {
                    if(!OrderModify(OrderTicket(), OrderOpenPrice(), newStop, OrderTakeProfit(), 0, clrRed))
                        Print("Failed to modify SELL order. Error: ", GetLastError());
                }
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
//| News Filter (Enhanced)                                           |
//+------------------------------------------------------------------+
bool IsNewsTime()
{
    // Enhanced news filter - check for major news times
    // This is a simplified version - in practice, integrate with news calendar
    
    int currentHour = Hour();
    int currentMinute = Minute();
    int currentDay = DayOfWeek();
    
    // Avoid trading during major news times (GMT)
    // US Non-Farm Payrolls (First Friday of month, 13:30 GMT)
    // FOMC meetings, ECB meetings, etc.
    
    // Example: Avoid trading 30 minutes before and after 13:30 GMT on Fridays
    if(currentDay == 5) // Friday
    {
        if((currentHour == 13 && currentMinute >= 0 && currentMinute <= 59) ||
           (currentHour == 14 && currentMinute >= 0 && currentMinute <= 30))
            return true;
    }
    
    // Add more news time filters as needed
    return false;
}

//+------------------------------------------------------------------+
//| Annotate Trade on Chart                                          |
//+------------------------------------------------------------------+
void AnnotateTrade(int trend, double price, int ticket)
{
    string label = (trend == 1) ? "BUY " : "SELL ";
    label += DoubleToStr(LotSize, 2) + " #" + IntegerToString(ticket);
    label += " @" + DoubleToStr(price, Digits);

    // Draw label at the trade price on the current bar
    string objName = "TradeLabel_" + IntegerToString(ticket);
    if(ObjectCreate(0, objName, OBJ_TEXT, 0, Time[0], price))
    {
        ObjectSetText(objName, label, TradeLabelFontSize, "Arial", TradeLabelColor);
        ObjectSetInteger(0, objName, OBJPROP_ANCHOR, ANCHOR_LEFT_UPPER);
    }

    // Draw a horizontal line at the trade price
    string lineName = "TradeLine_" + IntegerToString(ticket);
    if(ObjectCreate(0, lineName, OBJ_HLINE, 0, 0, price))
    {
        ObjectSetInteger(0, lineName, OBJPROP_COLOR, TradeLineColor);
        ObjectSetInteger(0, lineName, OBJPROP_WIDTH, 1);
        ObjectSetInteger(0, lineName, OBJPROP_STYLE, STYLE_DOT);
    }
}

//+------------------------------------------------------------------+
//| Deinit: Clean up chart objects                                   |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    Print("EA deinitialized. Reason: ", reason);
    
    // Remove all trade labels and lines
    for(int i=ObjectsTotal()-1; i>=0; i--)
    {
        string name = ObjectName(i);
        if(StringFind(name, "TradeLabel_") == 0 || StringFind(name, "TradeLine_") == 0)
            ObjectDelete(name);
    }
}

//+------------------------------------------------------------------+