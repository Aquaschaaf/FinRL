from finrl import config_private
import requests
import json
import logging


def activate_order(id):
    request = requests.post(config_private.PAPER_TRADING_ACT_URL.replace("{order_id}", str(id)),
                            data=json.dumps({
                                "pin": "7652"
                            }),
                            headers={"Authorization": f"Bearer {config_private.LEMON_API_KEY_PAPER}"})
    print(request.json())
    logging.info(request.json())



def place_order(isin, side, quantity, expiration="1D", venue="XMUN"):
    # Place inactive order
    request = requests.post(config_private.PAPER_TRADING_URL,
              data=json.dumps({
                  "isin": isin,
                  "expires_at": expiration,
                  "side": side,
                  "quantity": quantity,
                  "venue": venue,
                }),
              headers={"Authorization": f"Bearer {config_private.LEMON_API_KEY_PAPER}"})
    print(request.json())
    logging.info(request.json())

def get_orders(id=None):

    if id is not None:
        url = config_private.PAPER_TRADING_ORDER_URL + "/{}".format(id)
    else:
        url = config_private.PAPER_TRADING_ORDER_URL

    request = requests.get(url,headers={"Authorization": f"Bearer {config_private.LEMON_API_KEY_PAPER}"})

    print(request.json())

def cancel_order(id):
    request = requests.delete(config_private.PAPER_TRADING_ORDER_URL + "/{}".format(id),
                              headers={"Authorization": f"Bearer {config_private.LEMON_API_KEY_PAPER}"})
    print(request.json())


# place_order(isin="US19260Q1076", side="buy", quantity=2)
# activate_order(ID)
get_orders()