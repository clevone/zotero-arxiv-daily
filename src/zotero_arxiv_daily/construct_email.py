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
    .title-en {{
      font-size: 18px;
      font-weight: 700;
      line-height: 1.35;
      margin: 0 0 5px 0;
    }}
    .title-zh {{
      font-size: 16px;
      font-weight: 600;
      line-height: 1.45;
      color: #374151;
      margin: 0 0 10px 0;
    }}
    .meta {{
      color: #4b5563;
      font-size: 13px;
      line-height: 1.5;
      margin-bottom: 6px;
    }}
    .summary {{
      margin-top: 10px;
      line-height: 1.6;
    }}
    .summary + .summary {{
      margin-top: 8px;
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
      margin-right: 8px;
      margin-bottom: 6px;
    }}
    .links a.secondary {{
      background: #4b5563;
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
      <h1>Daily Literature Digest / 每日文献速递</h1>
      <p>Generated from your Zotero library and manually added focus topics. / 根据你的 Zotero 文献库和手动关注主题生成。</p>
    </div>
    __CONTENT__
    <div class="footer">
      To unsubscribe, remove your email in your GitHub Actions setting. / 如需停止接收，请在 GitHub Actions 设置中移除邮箱。
    </div>
  </div>
</body>
</html>
"""


def get_empty_html():
    return '<div class="empty">No Papers Today. Take a Rest! / 今日暂无论文，休息一下吧。</div>'


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


def _build_links_html(
    *,
    source: str | None,
    article_url: str | None,
    pdf_url: str | None,
) -> str:
    links = []
    normalized_source = (source or "").lower()

    # PubMed entries are bibliographic records; do not label publisher/DOI links
    # as "PDF". For PubMed, only show the stable PubMed record page.
    if normalized_source == "pubmed":
        if article_url:
            links.append(
                f'<a class="secondary" href="{_escape(article_url)}">PubMed / 文章页</a>'
            )
        return "".join(links)

    # For preprints, show both the abstract/article page and a real PDF link when available.
    if article_url:
        links.append(
            f'<a class="secondary" href="{_escape(article_url)}">Article page / 文章页</a>'
        )

    if pdf_url:
        links.append(
            f'<a href="{_escape(pdf_url)}">PDF</a>'
        )

    return "".join(links)


def get_block_html(
    title: str,
    authors: str,
    rate: str,
    tldr: str,
    pdf_url: str | None,
    affiliations: str = None,
    source: str | None = None,
    title_zh: str | None = None,
    tldr_zh: str | None = None,
    article_url: str | None = None,
):
    """
    Backward-compatible signature:
    old tests still call get_block_html(title, authors, rate, tldr, pdf_url, affiliations).
    """
    source_html = (
        f'<div class="badge">{_escape(_source_label(source))}</div>'
        if source
        else ""
    )
    title_zh_html = (
        f'<div class="title-zh">{_escape(title_zh)}</div>'
        if title_zh
        else ""
    )
    tldr_zh_html = (
        f'<div class="summary"><b>中文摘要：</b>{_escape(tldr_zh)}</div>'
        if tldr_zh
        else ""
    )
    links_html = _build_links_html(
        source=source,
        article_url=article_url,
        pdf_url=pdf_url,
    )

    # Compatibility for the old direct get_block_html() test:
    # if no source/article_url is supplied, preserve the simple PDF link behavior.
    if not links_html and pdf_url:
        links_html = f'<a href="{_escape(pdf_url)}">PDF</a>'

    block_template = """
    <div class="card">
      {source_html}
      <div class="title-en">{title}</div>
      {title_zh_html}
      <div class="meta">{authors}</div>
      <div class="meta">{affiliations}</div>
      <div class="meta">Relevance / 相关性：{rate}</div>
      <div class="summary"><b>TLDR (EN)：</b>{tldr}</div>
      {tldr_zh_html}
      <div class="links">{links_html}</div>
    </div>
    """

    return block_template.format(
        source_html=source_html,
        title=_escape(title),
        title_zh_html=title_zh_html,
        authors=_escape(authors),
        affiliations=_escape(affiliations or "Unknown Affiliation"),
        rate=_escape(str(rate)),
        tldr=_escape(tldr or ""),
        tldr_zh_html=tldr_zh_html,
        links_html=links_html,
    )


def get_stars(score: float):
    full_star = '<span class="full-star">⭐</span>'
    half_star = '<span class="half-star">⭐</span>'
    low = 6
    high = 8

    if score <= low:
        return ""
    if score >= high:
        return full_star * 5

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
    author_list = list(paper.authors)
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
                title=paper.title,
                authors=_format_authors(paper),
                rate=rate,
                tldr=paper.tldr_en or paper.tldr,
                pdf_url=paper.pdf_url,
                affiliations=_format_affiliations(paper),
                source=paper.source,
                title_zh=paper.title_zh,
                tldr_zh=paper.tldr_zh,
                article_url=paper.url,
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
    - render_email(papers) keeps the old single-list behavior;
    - render_email(preprints, published) renders the two bilingual sections.
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
                "今日预印本 / Today's Preprints",
                "优先展示当天新预印本；若当天不足 3 篇，则自动从未发送缓存池和近 30 天回补池中补齐。 / "
                "Prioritizes today's new preprints; if fewer than 3 are available, unsent cache and recent backfill are used.",
                papers,
            ),
            _render_section(
                "已发表精选 / Published Picks",
                "来自 PubMed 在线检索，并经高水平期刊白名单、超声相关性过滤和 Zotero 相关性排序筛选。 / "
                "Retrieved online from PubMed, then filtered by journal whitelist, ultrasound relevance, and Zotero-based ranking.",
                published_papers,
            ),
        ]
    )

    return framework.replace("__CONTENT__", content)
