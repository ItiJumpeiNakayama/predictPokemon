from tensorflow.keras import models
import numpy as np
# for using PIL, we have to add "Pillow" to requirements.txt
from PIL import Image
import io
from flask import jsonify, make_response
import json
import base64
import cv2 as cv
from google.cloud import storage

# Xception Fine Tuning モデルを読み込む（ Global variable として定義）
def loadCustomModel():
    # 'gs://kagawa-ai-lesson/model_4.hdf5'
    storage_client = storage.Client()
    bucket = storage_client.get_bucket('kagawa-ai-lesson')
    #blob = bucket.blob('model_4.hdf5')
    blob = bucket.blob('model_8.hdf5')
    blob.download_to_filename('/tmp/tmp.hdf5')
    return models.load_model('/tmp/tmp.hdf5')

# Xception Fine Tuning モデルを読み込む（ Global variable として定義）
model = loadCustomModel()

# ポケモン画像を予測する（ HTTP トリガーで呼び出されるメソッド）
def predictPokemon(request):
    """Responds to any HTTP request.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        The response text or any set of values that can be turned into a
        Response object using
        `make_response <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>`.
    """

# Set CORS headers for the preflight request
    if request.method == 'OPTIONS':
        # Allows GET requests from any origin with the Content-Type
        # header and caches preflight response for an 3600s
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }

        return ('', 204, headers)

    #if not request.files['captured']:
    #    data = dict(success=False, message='request.file["captured"] were not found.')
    #    return jsonify(data)

    #img = request.files['captured'].read()
    #img = Image.open(io.BytesIO(img))
    #img = img.resize((196, 196))
    #img.save('captured.jpg')

    request_json = request.get_json()
    captured = request_json['captured']
    image_decoded = base64.b64decode(captured.split(',')[1])  # remove header
    #image_obj = Image.open(io.BytesIO(image_decoded))
    #image_array = np.asarray(image_obj)
    #image_array = np.delete(image_array, 3, 2) # (196, 196, 4) -> (196, 196, 3)
    
    temp_array = np.fromstring(image_decoded, np.uint8)
    image_obj = cv.imdecode(temp_array, cv.IMREAD_COLOR)
    image_array = np.array(image_obj).reshape(196, 196, 3)

    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    print('#### image_array shape')
    print(image_array.shape)
    pred = model.predict(image_array)

    pokemon_names = [
          "Abra",
          "Aerodactyl",
          "Alakazam",
          "Arbok",
          "Arcanine",
          "Articuno",
          "Beedrill",
          "Bellsprout",
          "Blastoise",
          "Bulbasaur",
          "Butterfree",
          "Caterpie",
          "Chansey",
          "Charizard",
          "Charmander",
          "Charmeleon",
          "Clefable",
          "Clefairy",
          "Cloyster",
          "Cubone",
          "Dewgong",
          "Diglett",
          "Ditto",
          "Dodrio",
          "Doduo",
          "Dragonair",
          "Dragonite",
          "Dratini",
          "Drowzee",
          "Dugtrio",
          "Eevee",
          "Ekans",
          "Electabuzz",
          "Electrode",
          "Exeggcute",
          "Exeggutor",
          "Farfetchd",
          "Fearow",
          "Flareon",
          "Gastly",
          "Gengar",
          "Geodude",
          "Gloom",
          "Golbat",
          "Goldeen",
          "Golduck",
          "Golem",
          "Graveler",
          "Grimer",
          "Growlithe",
          "Gyarados",
          "Haunter",
          "Hitmonchan",
          "Hitmonlee",
          "Horsea",
          "Hypno",
          "Ivysaur",
          "Jigglypuff",
          "Jolteon",
          "Jynx",
          "Kabuto",
          "Kabutops",
          "Kadabra",
          "Kakuna",
          "Kangaskhan",
          "Kingler",
          "Koffing",
          "Krabby",
          "Lapras",
          "Lickitung",
          "Machamp",
          "Machoke",
          "Machop",
          "Magikarp",
          "Magmar",
          "Magnemite",
          "Magneton",
          "Mankey",
          "Marowak",
          "Meowth",
          "Metapod",
          "Mew",
          "Mewtwo",
          "Moltres",
          "MrMime",
          "Muk",
          "Nidoking",
          "Nidoqueen",
          "Nidorina",
          "Nidorino",
          "Ninetales",
          "Oddish",
          "Omanyte",
          "Omastar",
          "Onix",
          "Paras",
          "Parasect",
          "Persian",
          "Pidgeot",
          "Pidgeotto",
          "Pidgey",
          "Pikachu",
          "Pinsir",
          "Poliwag",
          "Poliwhirl",
          "Poliwrath",
          "Ponyta",
          "Porygon",
          "Primeape",
          "Psyduck",
          "Raichu",
          "Rapidash",
          "Raticate",
          "Rattata",
          "Rhydon",
          "Rhyhorn",
          "Sandshrew",
          "Sandslash",
          "Scyther",
          "Seadra",
          "Seaking",
          "Seel",
          "Shellder",
          "Slowbro",
          "Slowpoke",
          "Snorlax",
          "Spearow",
          "Squirtle",
          "Starmie",
          "Staryu",
          "Tangela",
          "Tauros",
          "Tentacool",
          "Tentacruel",
          "Vaporeon",
          "Venomoth",
          "Venonat",
          "Venusaur",
          "Victreebel",
          "Vileplume",
          "Voltorb",
          "Vulpix",
          "Wartortle",
          "Weedle",
          "Weepinbell",
          "Weezing",
          "Wigglytuff",
          "Zapdos",
          "Zubat"
    ]

    pokemon_names_jp = {
          "Abra": "ケーシィ",
          "Aerodactyl": "プテラ",
          "Alakazam": "フーディン",
          "Arbok": "アーボック",
          "Arcanine": "ウインディ",
          "Articuno": "フリーザー",
          "Beedrill": "スピアー",
          "Bellsprout": "マダツボミ",
          "Blastoise": "カメックス",
          "Bulbasaur": "フシギダネ",
          "Butterfree": "バタフリー",
          "Caterpie": "キャタピー",
          "Chansey": "ラッキー",
          "Charizard": "リザードン",
          "Charmander": "ヒトカゲ",
          "Charmeleon": "リザード",
          "Clefable": "ピクシー",
          "Clefairy": "ピッピ",
          "Cloyster": "パルシェン",
          "Cubone": "カラカラ",
          "Dewgong": "ジュゴン",
          "Diglett": "ディグダ",
          "Ditto": "メタモン",
          "Dodrio": "ドードリオ",
          "Doduo": "ドードー",
          "Dragonair": "ハクリュー",
          "Dragonite": "カイリュー",
          "Dratini": "ミニリュウ",
          "Drowzee": "スリープ",
          "Dugtrio": "ダグトリオ",
          "Eevee": "イーブイ",
          "Ekans": "アーボ",
          "Electabuzz": "エレブー",
          "Electrode": "マルマイン",
          "Exeggcute": "タマタマ",
          "Exeggutor": "ナッシー",
          "Farfetchd": "カモネギ",
          "Fearow": "オニドリル",
          "Flareon": "ブースター",
          "Gastly": "ゴース",
          "Gengar": "ゲンガー",
          "Geodude": "イシツブテ",
          "Gloom": "クサイハナ",
          "Golbat": "ゴルバット",
          "Goldeen": "トサキント",
          "Golduck": "ゴルダック",
          "Golem": "ゴローニャ",
          "Graveler": "ゴローン",
          "Grimer": "ベトベター",
          "Growlithe": "ガーディ",
          "Gyarados": "ギャラドス",
          "Haunter": "ゴースト",
          "Hitmonchan": "エビワラー",
          "Hitmonlee": "サワムラー",
          "Horsea": "タッツー",
          "Hypno": "スリーパー",
          "Ivysaur": "フシギソウ",
          "Jigglypuff": "プリン",
          "Jolteon": "サンダース",
          "Jynx": "ルージュラ",
          "Kabuto": "カブト",
          "Kabutops": "カブトプス",
          "Kadabra": "ユンゲラー",
          "Kakuna": "コクーン",
          "Kangaskhan": "ガルーラ",
          "Kingler": "キングラー",
          "Koffing": "ドガース",
          "Krabby": "クラブ",
          "Lapras": "ラプラス",
          "Lickitung": "ベロリンガ",
          "Machamp": "カイリキー",
          "Machoke": "ゴーリキー",
          "Machop": "ワンリキー",
          "Magikarp": "コイキング",
          "Magmar": "ブーバー",
          "Magnemite": "コイル",
          "Magneton": "レアコイル",
          "Mankey": "マンキー",
          "Marowak": "ガラガラ",
          "Meowth": "ニャース",
          "Metapod": "トランセル",
          "Mew": "ミュウ",
          "Mewtwo": "ミュウツー",
          "Moltres": "ファイヤー",
          "MrMime": "バリヤード",
          "Muk": "ベトベトン",
          "Nidoking": "ニドキング",
          "Nidoqueen": "ニドクイン",
          "Nidorina": "ニドリーナ",
          "Nidorino": "ニドリーノ",
          "Ninetales": "キュウコン",
          "Oddish": "ナゾノクサ",
          "Omanyte": "オムナイト",
          "Omastar": "オムスター",
          "Onix": "イワーク",
          "Paras": "パラス",
          "Parasect": "パラセクト",
          "Persian": "ペルシアン",
          "Pidgeot": "ピジョット",
          "Pidgeotto": "ピジョン",
          "Pidgey": "ポッポ",
          "Pikachu": "ピカチュウ",
          "Pinsir": "カイロス",
          "Poliwag": "ニョロモ",
          "Poliwhirl": "ニョロゾ",
          "Poliwrath": "ニョロボン",
          "Ponyta": "ポニータ",
          "Porygon": "ポリゴン",
          "Primeape": "オコリザル",
          "Psyduck": "コダック",
          "Raichu": "ライチュウ",
          "Rapidash": "ギャロップ",
          "Raticate": "ラッタ",
          "Rattata": "コラッタ",
          "Rhydon": "サイドン",
          "Rhyhorn": "サイホーン",
          "Sandshrew": "サンド",
          "Sandslash": "サンドパン",
          "Scyther": "ストライク",
          "Seadra": "シードラ",
          "Seaking": "アズマオウ",
          "Seel": "パウワウ",
          "Shellder": "シェルダー",
          "Slowbro": "ヤドラン",
          "Slowpoke": "ヤドン",
          "Snorlax": "カビゴン",
          "Spearow": "オニスズメ",
          "Squirtle": "ゼニガメ",
          "Starmie": "スターミー",
          "Staryu": "ヒトデマン",
          "Tangela": "モンジャラ",
          "Tauros": "ケンタロス",
          "Tentacool": "メノクラゲ",
          "Tentacruel": "ドククラゲ",
          "Vaporeon": "シャワーズ",
          "Venomoth": "モルフォン",
          "Venonat": "コンパン",
          "Venusaur": "フシギバナ",
          "Victreebel": "ウツボット",
          "Vileplume": "ラフレシア",
          "Voltorb": "ビリリダマ",
          "Vulpix": "ロコン",
          "Wartortle": "カメール",
          "Weedle": "ビードル",
          "Weepinbell": "ウツドン",
          "Weezing": "マタドガス",
          "Wigglytuff": "プクリン",
          "Zapdos": "サンダー",
          "Zubat": "ズバット"
    }

    confidence = str(round(max(pred[0]), 3))
    predicted_pokemon_name = pokemon_names[np.argmax(pred)]
    # Translate pokemon name to japanese
    predicted_pokemon_name_jp = pokemon_names_jp[predicted_pokemon_name]

    data = dict(success=True, predicted=predicted_pokemon_name, predicted_jp=predicted_pokemon_name_jp, confidence=confidence)
    ## return jsonify(data)
    #response = make_response(json.dumps(data, ensure_ascii=False))
    #response.headers['Access-Control-Allow-Origin'] = 'https://kagawa-ai-lesson.firebaseapp.com, http://localhost:8080'
    #response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, PATCH, DELETE, OPTIONS'
    #response.headers['Access-Control-Allow-Headers'] = 'DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range,Authorization'
    #response.headers['Access-Control-Expose-Headers'] = 'Content-Length,Content-Range'
    #return response

    # Set CORS headers for the main request
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Content-Type': 'application/json'
    }

    return (json.dumps(data, ensure_ascii=False), 200, headers)
