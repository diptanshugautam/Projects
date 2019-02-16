stk <- data$Sensex
int <- data$Interest
inf <- data$Inflation
remove(i1nt)
acf(int, lag.max = 10, type = c("correlation"))
pacf(int, lag.max = 10)
adf2 <- ur.df(inf, type = c("trend"), lags = 10, selectlags = c("BIC"))
summary(adf2)

ndiffs(stk, alpha = 0.05, test = c("adf","kpss"), max.d = 2)
ndiffs(int, alpha = 0.05, test = c("adf"), max.d = 2)
ndiffs(inf, alpha = 0.05, test = c("adf"), max.d = 2)
ndiffs(inf, alpha = 0.05, test = c("kpss"),max.d = 2)
diff_int <- rep(NA, 239)
for(i in 1:239){
  diff_int[i] = int[i+1]-int[i]
}
diff_stk <- rep(NA, 239)
for(j in 1:239){
  diff_stk[j] = stk[j+1]-stk[j]
}
plot(diff_stk, main ='Scatter plot of differenced Stock with days', xlab = 'No. of days',ylab = 'differenced Stock')
plot(diff_int, main ='Scatter plot of differenced interest rate with days', xlab = 'No. of days',ylab = 'differenced interest rate')
acf_summ <- acf(diff_int, lag.max = 10, type = ("correlation"))
coef(acf_summ)
summary(acf_summ)
pacf_summ <- pacf(diff_int, lag.max = 10)
coef(pacf_summ)
summary(pacf_summ)
acf(diff_stk, lag.max = 10, type = ("correlation"))
pacf(diff_stk, lag.max = 10)
acf(inf, lag.max = 10, type = ("correlation"))
pacf(inf, lag.max = 10)
arima_fit <- arima(x = diff_int, order = c(2L,1L,2L))
summary(arima_fit)
coef(arima_fit)
arima_fit <- auto.arima(int,max.p = 3,max.q = 3)
summary(arima_fit)
coef(arima_fit)
arima_fit1 <- auto.arima(stk,max.p = 3,max.q = 3)
summary(arima_fit)
coef(arima_fit)

library(SDSFoundations)
bdi_mult <- lm(stk~int+inf)
summary(bdi_mult)
coef(bdi_mult)
?svar
??svar
bdi_uni <- lm(inf~int)
summary(bdi_uni)
