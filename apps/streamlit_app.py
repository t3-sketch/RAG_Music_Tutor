import os
import sys
import tempfile
from pathlib import Path

# Streamlit Community Cloud には editable install が無いため、src layout の
# パッケージを直接解決できるようにする（editable install 済みのローカルでは no-op）
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st
from music_rag.query_pipeline import answer_query
from music_rag.audio_source import AudioSourceError
from music_rag import config

st.set_page_config(page_title="music-rag", page_icon="🎵")
st.title("🎵 music-rag")
st.caption("音楽理論の教材を根拠に、出典つきで質問へ答える RAG デモ")

with st.sidebar:
    top_k = st.slider("取得チャンク数 (top_k)", 1, 10, config.TOP_K)
    audio_file = None
    audio_url = ""
    if config.ENABLE_AUDIO_INPUT:
        audio_file = st.file_uploader(
            "音声ファイル（任意）", type=["mp3", "wav", "m4a", "flac", "ogg"],
            help="アップロードするとテンポ・キー・コード進行を解析し、回答の根拠に使います",
        )
        if config.ENABLE_URL_INPUT:
            audio_url = st.text_input(
                "音声URL（任意）", placeholder="https://www.youtube.com/watch?v=…",
                help="YouTube / ニコニコ動画のURL。ファイルと両方指定した場合はファイルを優先します",
            )

query = st.text_input("質問を入力", placeholder="例: ドミナントモーションとは？")

if st.button("質問する", type="primary") and query.strip():
    # librosa はファイルパスを期待するため、アップロードを一時ファイルに書き出す。
    # URL の場合はそのまま渡し、ダウンロードと後始末は audio_source に任せる。
    audio_path = None
    upload_tmp_path = None
    if audio_file is not None:
        suffix = os.path.splitext(audio_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_file.getbuffer())
            audio_path = upload_tmp_path = tmp.name
    elif audio_url.strip():
        audio_path = audio_url.strip()

    try:
        with st.spinner("検索・生成中…（初回はモデル読込で時間がかかります）"):
            result = answer_query(query.strip(), top_k=top_k,
                                  audio_path=audio_path)
    except (AudioSourceError, ValueError) as e:
        st.error(str(e))
        st.stop()
    finally:
        if upload_tmp_path:
            os.unlink(upload_tmp_path)

    st.markdown("### 回答")
    st.write(result["answer"])
    if result.get("audio_desc"):
        with st.expander("音響解析結果（テンポ・キー・コード進行）"):
            st.text(result["audio_desc"])
    st.markdown("### 出典")
    seen_sources = set()
    for c in result["contexts"]:
        if c["source"] in seen_sources:
            continue
        seen_sources.add(c["source"])
        if c.get("source_url"):
            # 出典メタ付きコーパス（OMT等）: 章タイトルへのリンクで表示
            st.markdown(f"- [{c.get('title') or c['source']}]({c['source_url']})")
        else:
            # 旧コーパス（メタなし）へのフォールバック
            st.markdown(f"- `{c['source']}`")

    # CC BY-SA等の帰属表示: contexts中のユニークな教材ごとにライセンスを明記
    books = {(c["book"], c["license"], c.get("license_url"))
             for c in result["contexts"] if c.get("book") and c.get("license")}
    for book, license_name, license_url in sorted(books):
        notice = f"出典教材: {book} — {license_name}"
        if license_url:
            notice += f" ({license_url})"
        st.caption(notice)

    if config.SHOW_DEBUG_CHUNKS:
        with st.expander("取得チャンク（デバッグ）"):
            for i, c in enumerate(result["contexts"], 1):
                st.markdown(f"**{i}. {c['source']}**（score {c['score']:.3f}）")
                st.text(c["text"][:200])
