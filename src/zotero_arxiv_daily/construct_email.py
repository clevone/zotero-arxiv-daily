from __future__ import annotations

import html
import math

from .protocol import Paper


framework = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <style>
    body {{
      font-family: Arial, Helvetica, sans-serif;
      background: #f6f7fb;
      color: #1f2937;
      margin: 0;
      padding: 24px 12px;
    }}
    .container {{
      max-width: 760px;
      margin: 0 auto;
    }}
    .header {{
      margin-bottom: 18px;
    }}
    .header h1 {{
      font-size: 24px;
      margin: 0 0 6px 0;
    }}
    .header p {{
      margin: 0;
      color: #6b7280;
      line-height: 1.5;
    }}
    .section {{
      margin: 22px 0 26px 0;
    }}
    .section-title {{
      font-size: 20px;
      margin: 0 0 4px 0;
    }}
    .section-note {{
      color: #6b7280;
      font-size: 13px;
      margin: 0 0 12px 0;
      line-height: 1.5;
    }}
    .card {{
      background: #ffffff;
      border-radius: 14px;
      padding: 18px 18px 16px 18px;
      margin: 12px 0;
      box-shadow: 0 1px 4px rgba(0, 0, 0, 0.08);
    }}
    .badge {{
      display: inline-block;
      font-size: 12px;
      line-height: 1;
      padding: 6px 8px;
      border-radius: 999px;
      background: #eef2ff;
      color: #3730a3;
      margin-bottom: 10px;
    }}
    .title {{
      font-size: 18px;
      font-weight: 700;
      line-height: 1.35;
      margin: 0 0 8px 0;
    }}
    .meta {{
      color: #4b5563;
      font-size: 13px;
      line-height: 1.5;
      margin-bottom: 6px;
    }}
    .tldr {{
      margin-top: 10px;
      line-height: 1.6;
    }}
    .links {{
      margin-top: 12px;
    }}
    .links a {{
      display: inline-block;
      text-decoration: none;
      padding: 8px 10px;
      border-radius: 8px;
      background: #ef4444;
      color: #ffffff;
      font-size: 13px;
    }}
    .empty {{
      background: #ffffff;
      border-radius: 14px;
      padding: 18px;
      color: #6b7280;
      box-shadow: 0 1px 4px rgba(0, 0, 0, 0.08);
    }}
    .footer {{
      margin-top: 28px;
      color: #9ca3af;
      font-size: 12px;
    }}
    .full-star, .half-star {{
      color: #f59e0b;
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>Daily Literature Digest</h1>
      <p>根据你的 Zotero 文献库兴趣画像生成。</p>
    </div>
    __CONTENT__
    <div class="footer">
      To unsubscribe, remove your email in your GitHub Actions setting.
    </div>
  </div>
</body>
</html>
"""


def get_empty_html():
    return '<div class="empty">No Papers Today. Take a Rest!</div>'


def _escape(value: str | None) -> str:
    return html.escape(value or "")


def _source_label(source: str | None) -> str:
    mapping = {
        "arxiv": "arXiv",
        "biorxiv": "bioRxiv",
        "medrxiv": "medRxiv",
        "pubmed": "PubMed",
    }
    return mapping.get((source or "").lower(), source or "Unknown")


def get_block_html(
    title: str,
    authors: str,
    rate: str,
    tldr: str,
    pdf_url: str,
    affiliations: str = None,
    source: str | None = None,
):
    """
    保持原来的函数名和参数顺序，保证旧测试兼容；
    新增 source 为可选参数，用于在邮件卡片里显示来源标签。
    """
    source_html = (
        f'<div class="badge">{_escape(_source_label(source))}</div>'
        if source
        else ""
    )

    block_template = """
    <div class="card">
      {source_html}
      <div class="title">{title}</div>
      <div class="meta">{authors}</div>
      <div class="meta">{affiliations}</div>
      <div class="meta">Relevance: {rate}</div>
      <div class="tldr"><b>TLDR:</b> {tldr}</div>
      <div class="links"><a href="{pdf_url}">PDF</a></div>
    </div>
    """

    return block_template.format(
        source_html=source_html,
        title=_escape(title),
        authors=_escape(authors),
        affiliations=_escape(affiliations or "Unknown Affiliation"),
        rate=_escape(str(rate)),
        tldr=_escape(tldr or ""),
        pdf_url=_escape(pdf_url or ""),
    )


def get_stars(score: float):
    full_star = '<span class="full-star">⭐</span>'
    half_star = '<span class="half-star">⭐</span>'
    low = 6
    high = 8

    if score <= low:
        return ""
    elif score >= high:
        return full_star * 5
    else:
        interval = (high - low) / 10
        star_num = math.ceil((score - low) / interval)
        full_star_num = int(star_num / 2)
        half_star_num = star_num - full_star_num * 2

        return (
            '<div class="star-wrapper">'
            + full_star * full_star_num
            + half_star * half_star_num
            + "</div>"
        )


def _format_authors(paper: Paper) -> str:
    author_list = [a for a in paper.authors]
    num_authors = len(author_list)

    if num_authors <= 5:
        return ", ".join(author_list)

    return ", ".join(author_list[:3] + ["..."] + author_list[-2:])


def _format_affiliations(paper: Paper) -> str:
    if paper.affiliations is None:
        return "Unknown Affiliation"

    affiliations = paper.affiliations[:5]
    text = ", ".join(affiliations)

    if len(paper.affiliations) > 5:
        text += ", ..."

    return text


def _render_paper_cards(papers: list[Paper]) -> str:
    parts = []

    for paper in papers:
        rate = round(paper.score, 1) if paper.score is not None else "Unknown"
        parts.append(
            get_block_html(
                paper.title,
                _format_authors(paper),
                rate,
                paper.tldr,
                paper.pdf_url,
                _format_affiliations(paper),
                source=paper.source,
            )
        )

    return "".join(parts)


def _render_section(title: str, note: str, papers: list[Paper]) -> str:
    body = _render_paper_cards(papers) if papers else get_empty_html()

    return f"""
    <div class="section">
      <h2 class="section-title">{_escape(title)}</h2>
      <p class="section-note">{_escape(note)}</p>
      {body}
    </div>
    """


def render_email(
    papers: list[Paper],
    published_papers: list[Paper] | None = None,
) -> str:
    """
    Backward compatible:
    - render_email(papers) 保持旧行为，测试仍然可用；
    - render_email(preprints, published) 生成两个板块：
      “今日预印本” 与 “已发表精选”。
    """
    if published_papers is None:
        if len(papers) == 0:
            return framework.replace("__CONTENT__", get_empty_html())

        return framework.replace("__CONTENT__", _render_paper_cards(papers))

    if len(papers) == 0 and len(published_papers) == 0:
        return framework.replace("__CONTENT__", get_empty_html())

    content = "".join(
        [
            _render_section(
                "今日预印本",
                "优先展示当天新预印本；若当天不足 3 篇，则自动从未发送缓存池和近 30 天回补池中补齐。",
                papers,
            ),
            _render_section(
                "已发表精选",
                "来自 PubMed 在线检索，并经高水平期刊白名单、超声相关性过滤和 Zotero 相关性排序筛选。",
                published_papers,
            ),
        ]
    )

    return framework.replace("__CONTENT__", content)
