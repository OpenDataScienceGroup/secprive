import sys
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(); pg.goto("file:///home/claude/proj/report.html")
    pg.pdf(path=sys.argv[1], format="A4", display_header_footer=True, header_template="<span></span>",
           footer_template="<div style='font-size:10pt;width:100%;text-align:center'><span class='pageNumber'></span></div>",
           margin=dict(top="2cm",bottom="2cm",left="2cm",right="2cm"))
    b.close()
