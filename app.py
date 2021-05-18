from __future__ import unicode_literals
import os
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, ImageMessage
from file import File

app = Flask(__name__)

@app.route('/')
def index():
    return 'Home Page'


# Channel Access Token
file = open('./channel_access_token.txt', encoding='utf-8')
text = file.read().strip()
line_bot_api = LineBotApi(text)
file.close()

# Channel Secret
file = open('./channel_secret.txt', encoding='utf-8')
text = file.read().strip()
handler = WebhookHandler(text)
file.close()


# 監聽所有來自 /callback 的 Post Request
@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']
    # get request body as text
    body = request.get_data(as_text=True)
    # app.logger.info("Request body: " + body)
    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'


# 處理訊息
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    message = TextSendMessage(text='哈囉你好，請傳給我一個坐姿正面照')
    line_bot_api.reply_message(event.reply_token, message)


@handler.add(MessageEvent, message=ImageMessage)
def handle_content_message(event):
    is_image = False
    if isinstance(event.message, ImageMessage):
        ext = 'jpg'
        is_image = True
    # elif isinstance(event.message, VideoMessage):
    #     ext = 'mp4'
    # elif isinstance(event.message, AudioMessage):
    #     ext = 'm4a'
    else:
        return

    if is_image is False:
        line_bot_api.reply_message(event.reply_token, '這好像不是圖片唷')
    else:
        message_content = line_bot_api.get_message_content(event.message.id)
        img, file_path = file.save_bytes_image(message_content.content)
        pred = ai.predict_image_with_path(file_path)
        line_bot_api.reply_message(
            event.reply_token, [
                TextSendMessage(text=pred)
            ])


@app.route('/')
def index():
    return 'Hello ttGroup'


ai = AI()
file = File()

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

