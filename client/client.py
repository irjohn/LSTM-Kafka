# IMPORTS

import base64
import hmac
import hashlib
import json
import aiohttp
import requests
import time
import asyncio
import math
from .exceptions import PhemexAPIException
from .config import Config
from termcolor import colored
from datetime import datetime
#---------------------------------------------------------------------->
def get_breakeven(data):
    from numpy import arange
    realizedPnL,avgEntry,positionQty = data
    openValue = (positionQty*1)/avgEntry
    if realizedPnL < 0:
        for n in arange(avgEntry,avgEntry*10,0.05):
            closeValue = (positionQty*1)/n
            PnL = openValue-closeValue
            if PnL >= abs(realizedPnL):
                return round(n,2)
    elif realizedPnL > 0:
        for n in arange(avgEntry,0,-0.05):
            closeValue = (positionQty*1)/n
            PnL = openValue-closeValue
            if abs(PnL) >= realizedPnL:
                return round(n,2)       
    else:
        return avgEntry

def find_average(qty,avg,newqty,price):
    return (qty*avg+newqty*price)/(newqty+qty)

#---------------------------------------------------------------------->
class PhemexClient(Config):
    MAIN_NET_API_URL = 'https://api.phemex.com'
    TEST_NET_API_URL = 'https://testnet-api.phemex.com'

    CURRENCY_BTC = "BTC"
    CURRENCY_USD = "USD"

    SYMBOL_BTCUSD = "BTCUSD"
    SYMBOL_ETHUSD = "ETHUSD"
    SYMBOL_XRPUSD = "XRPUSD"

    SIDE_BUY = "Buy"
    SIDE_SELL = "Sell"

    ORDER_TYPE_MARKET = "Market"
    ORDER_TYPE_LIMIT = "Limit"

    TIF_IMMEDIATE_OR_CANCEL = "ImmediateOrCancel"
    TIF_GOOD_TILL_CANCEL = "GoodTillCancel"
    TIF_FOK = "FillOrKill"

    ORDER_STATUS_NEW = "New"
    ORDER_STATUS_PFILL = "PartiallyFilled"
    ORDER_STATUS_FILL = "Filled"
    ORDER_STATUS_CANCELED = "Canceled"
    ORDER_STATUS_REJECTED = "Rejected" 
    ORDER_STATUS_TRIGGERED = "Triggered"
    ORDER_STATUS_UNTRIGGERED = "Untriggered"

    def __init__(self,is_testnet=False):
        Config.__init__(self)
        self.api_key = self.id
        self.api_secret = self.secret
        self.api_URL = self.MAIN_NET_API_URL
        if is_testnet:
            self.api_URL = self.TEST_NET_API_URL
        self.session = requests.session()
        self.set_positionInfo()

    def round05(cls,n):
        return (math.ceil(n * 20)/20)


    def set_positionInfo(self):
        self.accountData = self.query_account_n_positions(self.asset)
        if self.accountData['code'] == 0:
            self.position_info = self.accountData['data']   
            self.accountBalance = self.position_info['account']['accountBalanceEv']/100000000
            self.markPrice = self.position_info['positions'][0]['markPrice']
            self.liquidationPrice = self.position_info['positions'][0]['liquidationPrice']
            self.positionMargin = self.position_info['positions'][0]['positionMargin']
            self.assignedMargin = self.position_info['positions'][0]['assignedPosBalance']
            self.usedBalance = self.position_info['positions'][0]['usedBalance']
            self.avgEntryPrice = self.position_info['positions'][0]['avgEntryPrice']
            self.positionQty = self.position_info['positions'][0]['size']
            self.positionValue = self.position_info['positions'][0]['value']
            self.margin = (1-(self.liquidationPrice/self.markPrice))*100
            if self.positionQty > 0:
                self.unrealizedPNL = (self.positionQty/1)/self.avgEntryPrice-((self.positionQty/1)/self.markPrice)
                self.realizedPNL = self.position_info['positions'][0]['curTermRealisedPnlEv']/self.Ev
                data = (self.realizedPNL,self.avgEntryPrice,self.positionQty)
                self.breakeven = get_breakeven(data)
            else:
                self.unrealizedPNL = 0
                self.realizedPNL = 0
                self.breakeven = 0


    def _send_request(self, method, endpoint, params={}, body={}):
        expiry = str(math.trunc(time.time()) + 60)
        query_string = '&'.join(['{}={}'.format(k,v) for k,v in params.items()])
        message = endpoint + query_string + expiry
        body_str = ""
        if body:
            body_str = json.dumps(body, separators=(',', ':'))
            message += body_str
        signature = hmac.new(self.api_secret.encode('utf-8'), message.encode('utf-8'), hashlib.sha256)
        self.session.headers.update({
            'x-phemex-request-signature': signature.hexdigest(),
            'x-phemex-request-expiry': expiry,
            'x-phemex-access-token': self.api_key,
            'Content-Type': 'application/json'})

        url = self.api_URL + endpoint
        if query_string:
            url += '?' + query_string
        response = self.session.request(method, url, data=body_str.encode())
        if not str(response.status_code).startswith('2'):
            print(response.text)
            raise PhemexAPIException(response)
        try:
            res_json = response.json()
        except ValueError:
            raise PhemexAPIException('Invalid Response: %s' % response.text)
        if "code" in res_json and res_json["code"] != 0:
            raise PhemexAPIException(response)
        if "error" in res_json and res_json["error"]:
            raise PhemexAPIException(response)
        return res_json

    async def __send_request(self, method, endpoint, params={}, body={}):
        expiry = str(math.trunc(time.time()) + 60)
        query_string = '&'.join(['{}={}'.format(k,v) for k,v in params.items()])
        message = endpoint + query_string + expiry
        body_str = ""
        if body:
            body_str = json.dumps(body, separators=(',', ':'))
            message += body_str
        signature = hmac.new(self.api_secret.encode('utf-8'), message.encode('utf-8'), hashlib.sha256)
        async with aiohttp.ClientSession() as session:
            session.headers.update({
                'x-phemex-request-signature': signature.hexdigest(),
                'x-phemex-request-expiry': expiry,
                'x-phemex-access-token': self.api_key,
                'Content-Type': 'application/json'})
            url = self.api_URL + endpoint
            if query_string:
                url += '?' + query_string
            async with session.request(method, url, data=body_str.encode()) as response:
                if not str(response.status).startswith('2'):
                    raise PhemexAPIException(response)
                try:
                    res_json = await response.json()
                except ValueError:
                    raise PhemexAPIException('Invalid Response: %s' % response.text)
                if "code" in res_json and res_json["code"] != 0:
                    raise PhemexAPIException(response)
                if "error" in res_json and res_json["error"]:
                    raise PhemexAPIException(response)
                return res_json

    async def _send_orders(self,orders):
        """
        performs asynchronous post requests
        """
        async def send_all(orders):
            async with aiohttp.ClientSession() as session:
                async def send_order(method='post',params={}, body={}):
                    endpoint = "/orders"
                    expiry = str(math.trunc(time.time()) + 60)
                    query_string = '&'.join(['{}={}'.format(k,v) for k,v in params.items()])
                    message = endpoint + query_string + expiry
                    body_str = ""
                    if body:
                        body_str = json.dumps(body, separators=(',', ':'))
                        message += body_str
                    signature = hmac.new(self.api_secret.encode('utf-8'), message.encode('utf-8'), hashlib.sha256)
                    session.headers.update({
                        'x-phemex-request-signature': signature.hexdigest(),
                        'x-phemex-request-expiry': expiry,
                        'x-phemex-access-token': self.api_key,
                        'Content-Type': 'application/json'})
                    url = self.api_URL + endpoint
                    if query_string:
                         url += '?' + query_string
                    async with session.request(method,url,data=body_str.encode()) as response:
                        res_json = await response.json()
                        #print(res_json)
                        #try:
                        #    res_json = await response.json()
                        #except ValueError:
                        #    raise PhemexAPIException('Invalid Response: %s' % response.text)
                        if not str(response.status).startswith('2'):
                            raise PhemexAPIException(response)    
                        if "code" in res_json and res_json["code"] != 0:
                            raise PhemexAPIException(response)
                        if "error" in res_json and res_json["error"]:
                            raise PhemexAPIException(response)
                        else:
                            price = res_json['data']['price']
                            if res_json['data']['side'] == 'Buy':
                                #self.updateHoldings(res_json)
                                return print(self.buy_header + f' {price}')
                            elif res_json['data']['side'] == 'Sell':
                                originalPrice = self.findNumber(res_json['data']['clOrdID'])
                                res_json['data']['originalPrice'] = originalPrice
                                #print(originalPrice)
                                #self.updateHoldings(res_json)
                                return print(self.sell_header + f' {originalPrice} | '+self.sellPrice_header+f' {price}')
                return await asyncio.gather(*[send_order(body=order) for order in orders],return_exceptions=True)
        return await send_all(orders)

    async def _cancel_orders(self,symbol,orderIDs):
        """
        performs asynchronous delete requests
        """
        async def cancel_all(orders):
            async with aiohttp.ClientSession() as session:
                async def cancel_order(method='delete',params={},body={}):
                    endpoint = "/orders/cancel"
                    expiry = str(math.trunc(time.time()) + 60)
                    query_string = '&'.join(['{}={}'.format(k,v) for k,v in params.items()])
                    message = endpoint + query_string + expiry
                    body_str = ""
                    if body:
                        body_str = json.dumps(body, separators=(',', ':'))
                        message += body_str
                    signature = hmac.new(self.api_secret.encode('utf-8'), message.encode('utf-8'), hashlib.sha256)
                    session.headers.update({
                        'x-phemex-request-signature': signature.hexdigest(),
                        'x-phemex-request-expiry': expiry,
                        'x-phemex-access-token': self.api_key,
                        'Content-Type': 'application/json'})
                    url = self.api_URL + endpoint
                    if query_string:
                         url += '?' + query_string
                    async with session.request(method,url,data=body_str.encode()) as response:
                        res_json = await response.json()
                        #print(res_json)
                        #try:
                        #    res_json = await response.json()
                        #except ValueError:
                        #    raise PhemexAPIException('Invalid Response: %s' % response.text)
                        if not str(response.status).startswith('2'):
                            raise PhemexAPIException(response) 
                        if "code" in res_json and res_json["code"] != 0:
                            raise PhemexAPIException(response)
                        if "error" in res_json and res_json["error"]:
                            raise PhemexAPIException(response)
                            if json_res['error']['code'] == 10002:
                                print(self.cancelError_header +f' {params["orderID"]}')
                                self.db.delete_order(params['orderID'])
                                return json_res
                        return print(self.cancel_header+f' {params["orderID"]}')
                return await asyncio.gather(*[cancel_order(params={'symbol':symbol,
                                                                   'orderID':orderID}) for orderID in orderIDs],return_exceptions=True)
        return await cancel_all(orderIDs)

    def query_account_n_positions(self, currency:str):
        """
        https://github.com/phemex/phemex-api-docs/blob/master/Public-API-en.md#querytradeaccount
        """
        return self._send_request("get", "/accounts/accountPositions", {'currency':currency})
    
    def query_accountBalance(self,params={}):
        """
        https://github.com/phemex/phemex-api-docs/blob/master/Public-API-en.md#querytradeaccount
        """
        return self._send_request("get", "/phemex-user/users/children",params=params)

    def query_closed_orders(self, params={}):
        """
        https://github.com/phemex/phemex-api-docs/blob/master/Public-API-en.md#querytradeaccount
        """
        return self._send_request("get", "/exchange/order/list", params=params)

    def query_open_orders(self, symbol):
        """
        https://github.com/phemex/phemex-api-docs/blob/master/Public-API-en.md#6210-query-open-orders-by-symbol
        """
        return self._send_request("GET", "/orders/activeList", params={"symbol": symbol})

    def query_kline(self, params={}):
        """
        https://github.com/phemex/phemex-api-docs/blob/master/Public-API-en.md#querytradeaccount
        """
        return self._send_request("get", "/exchange/public/md/kline", params=params)

    def place_order(self, params={}):
        """
        https://github.com/phemex/phemex-api-docs/blob/master/Public-API-en.md#placeorder
        """
        return self._send_request("post", "/orders", body=params)

    def amend_order(self, symbol, orderID, params={}):
        """
        https://github.com/phemex/phemex-api-docs/blob/master/Public-API-en.md#622-amend-order-by-orderid
        """
        params["symbol"] = symbol
        params["orderID"] = orderID
        return self._send_request("put", "/orders/replace", params=params)

    def cancel_order(self, symbol, orderID):
        """
        https://github.com/phemex/phemex-api-docs/blob/master/Public-API-en.md#623-cancel-single-order
        """
        return self._send_request("delete", "/orders/cancel", params={"symbol": symbol, "orderID": orderID})

    def _cancel_all(self, symbol, untriggered_order=False):
        """
        https://github.com/phemex/phemex-api-docs/blob/master/Public-API-en.md#625-cancel-all-orders
        """
        return self._send_request("delete", "/orders/all", 
            params={"symbol": symbol, "untriggered": str(untriggered_order).lower()})
    
    def cancel_all_normal_orders(self, symbol):
        self._cancel_all(symbol, untriggered_order=False)

    def cancel_all_untriggered_conditional_orders(self, symbol):
        self._cancel_all(symbol, untriggered_order=True)

    def cancel_all_orders(self, symbol):
        self._cancel_all(symbol, untriggered_order=False)
        self._cancel_all(symbol, untriggered_order=True)

    def change_leverage(self, symbol, leverage=0):
        """
        https://github.com/phemex/phemex-api-docs/blob/master/Public-API-en.md#627-change-leverage
        """
        return self._send_request("PUT", "/positions/leverage", params={"symbol":symbol, "leverage": leverage})

    def change_risklimit(self, symbol, risk_limit=0):
        """
        https://github.com/phemex/phemex-api-docs/blob/master/Public-API-en.md#628-change-position-risklimit
        """
        return self._send_request("PUT", "/positions/riskLimit", params={"symbol":symbol, "riskLimit": risk_limit})

    def query_24h_ticker(self, symbol):
        """
        https://github.com/phemex/phemex-api-docs/blob/master/Public-API-en.md#633-query-24-hours-ticker
        """
        return self._send_request("GET", "/md/ticker/24hr", params={"symbol": symbol})

if __name__ == '__main__':
    client = PhemexClient()
