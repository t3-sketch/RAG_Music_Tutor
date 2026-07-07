"""音声入力ソースの解決レイヤー（URL → ローカル一時ファイルパス）。

「下流はすべてローカルファイルパスを話す」という audio_path 契約の手前に置き、
YouTube / ニコニコ動画の URL をローカル一時ファイルに解決する。
ローカルパスが渡された場合はそのまま返す（下流の audio.py は一切変更不要）。

yt-dlp によるダウンロードは各サービスの利用規約に抵触しうるため、
ローカル個人利用・研究目的限定の機能（config.ENABLE_URL_INPUT で無効化可能）。
ニコニコのログイン必須・プレミアム限定動画は非対応（cookie 連携はスコープ外）。
"""
import os
import shutil
import tempfile
from urllib.parse import urlparse

from music_rag import config

# 対象サービス以外の任意 URL をダウンロードしない（SSRF 的な悪用の防止も兼ねる）
_ALLOWED_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "music.youtube.com",
    "m.youtube.com",
    "youtu.be",
    "nicovideo.jp",
    "www.nicovideo.jp",
    "sp.nicovideo.jp",
    "nico.ms",
}

# resolve() が作る一時ディレクトリの目印。cleanup() はこれを見て削除可否を判断する。
_TMP_PREFIX = "music_rag_dl_"


class AudioSourceError(RuntimeError):
    """URL 解決の失敗（ユーザーにそのまま見せられる日本語メッセージを持つ）。"""


def _is_url(source: str) -> bool:
    return urlparse(source).scheme in ("http", "https")


def _to_user_message(error_text: str) -> str:
    """yt-dlp のエラー文をユーザー向けの日本語メッセージに変換する。"""
    lowered = error_text.lower()
    if "private" in lowered:
        return "この動画は非公開のためダウンロードできません。"
    if "unavailable" in lowered or "removed" in lowered or "deleted" in lowered:
        return "この動画は削除済みか、利用できない状態です。"
    if "age" in lowered and ("confirm" in lowered or "restrict" in lowered):
        return "年齢制限付きの動画には対応していません。"
    if "login" in lowered or "premium" in lowered or "member" in lowered:
        return "ログイン必須・会員限定の動画には対応していません。"
    if "not available in your country" in lowered or "region" in lowered or "geo" in lowered:
        return "お住まいの地域では利用できない動画です。"
    return f"音声のダウンロードに失敗しました（yt-dlp の更新で直る場合があります）: {error_text}"


def resolve(source: str) -> str:
    """URL ならダウンロードして一時ファイルのパスを、ローカルパスならそのまま返す。

    Raises:
        ValueError: 対象外ドメインの URL。
        AudioSourceError: 動画長の超過、ダウンロード失敗。
    """
    if not _is_url(source):
        return source

    # UI 側で入力欄を隠すだけでは event 経由の直接投入を防げないため、ここでも弾く
    if not config.ENABLE_URL_INPUT:
        raise AudioSourceError(
            "音声URL入力は無効化されています（ENABLE_URL_INPUT を確認してください）。"
        )

    host = urlparse(source).hostname or ""
    if host not in _ALLOWED_HOSTS:
        raise ValueError(
            f"対応していないURLです（YouTube / ニコニコ動画のみ対応）: {host}"
        )

    import yt_dlp  # 重い依存なので使う時だけ import（BTC の lazy import と同じ方針）

    tmpdir = tempfile.mkdtemp(prefix=_TMP_PREFIX)
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(tmpdir, "%(id)s.%(ext)s"),
        "postprocessors": [
            # mp3: soundfile(libsndfile)がネイティブ対応 → librosa側のaudioreadフォールバック警告を回避
            {"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}
        ],
        # -ac 1: モノラルにダウンミックスして容量削減。
        # 下流の librosa.load も mono=True で読むため、解析結果への影響はない。
        "postprocessor_args": {"ffmpeg": ["-ac", "1"]},
        "noplaylist": True,
        "quiet": True,
        "noprogress": True,  # quiet だけでは進捗バーは消えない（yt-dlp の仕様）
        "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # まずメタデータのみ取得し、動画長で弾く（ダウンロード帯域と BTC 推論時間の節約）
            info = ydl.extract_info(source, download=False)
            duration = info.get("duration")
            if duration and duration > config.MAX_AUDIO_DURATION_SEC:
                raise AudioSourceError(
                    f"動画が長すぎます（{int(duration)}秒 > 上限{config.MAX_AUDIO_DURATION_SEC}秒）。"
                    "解析時間の都合により、上限以内の動画を指定してください。"
                )
            ydl.download([source])
            path = os.path.join(tmpdir, f"{info['id']}.mp3")
            if not os.path.exists(path):
                raise AudioSourceError(
                    "ダウンロード後の音声ファイルが見つかりません（ffmpeg の導入を確認してください）。"
                )
            return path
    except yt_dlp.utils.DownloadError as e:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise AudioSourceError(_to_user_message(str(e))) from e
    except Exception:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise


def cleanup(path: str, source: str) -> None:
    """resolve() がダウンロードしたファイルだけ削除する（元がローカルパスなら何もしない）。"""
    if not _is_url(source):
        return
    parent = os.path.dirname(path)
    if os.path.basename(parent).startswith(_TMP_PREFIX):
        shutil.rmtree(parent, ignore_errors=True)
