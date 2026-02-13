# Parsing Markdown to PDF

## Prerequisites

- VS Code extension: **Markdown PDF** (`yzane.markdown-pdf`) - for documents
- VS Code extension: **Marp for VS Code** (`marp-team.marp-vscode`) - for presentations

## Setup

### 1. `.vscode/` folder

Place the `.vscode/` folder at the **workspace root** (the folder you open in VS Code):

```
<workspace-root>/
  .vscode/
    settings.json
    doc-style.css
```

### 2. `settings.json`

Contains the Markdown PDF export configuration:

```json
{
  "markdown-pdf.headerTemplate": "<div></div>",
  "markdown-pdf.footerTemplate": "<div style=\"font-size:9pt; font-family:'Times New Roman'; width:100%; text-align:center; margin:0 auto;\"><span class='pageNumber'></span></div>",
  "markdown-pdf.displayHeaderFooter": true,
  "markdown-pdf.format": "A4",
  "markdown-pdf.scale": 0.85,
  "markdown-pdf.margin.top": "22mm",
  "markdown-pdf.margin.bottom": "22mm",
  "markdown-pdf.margin.left": "20mm",
  "markdown-pdf.margin.right": "20mm",
  "markdown-pdf.styles": ["./.vscode/doc-style.css"]
}
```

- **Page**: A4, 85% scale, 22mm top/bottom and 20mm left/right margins
- **Header**: hidden
- **Footer**: centered page number in Times New Roman 9pt
- **Styles**: references `doc-style.css` from the `.vscode/` folder (path is relative to workspace root)

### 3. `doc-style.css`

Must be moved from the current directory to the `.vscode/` folder at the workspace root. Custom stylesheet applied to the PDF output:

- **Body**: Times New Roman 12pt, 1.15 line height
- **Headings**: Times New Roman (h1: 16pt, h2: 14pt, h3: 13pt)
- **Code/Pre**: Courier New 9pt
- **Tables**: full-width, 11pt, 1px solid borders

### 4. Export

#### Document

1. Open a `*_documentation.md` file in VS Code
2. `Ctrl+Shift+P` > **Markdown PDF: Export (pdf)**
3. The PDF is generated in the same directory as the `.md` file

#### Presentation

Presentation files use **Marp** (identified by `marp: true` in the YAML front matter). Slides are separated by `---`.

1. Open a `*_presentation.md` file in VS Code
2. `Ctrl+Shift+P` > **Marp: Export Slide Deck...**
3. Select **PDF** as the output format
4. The PDF is generated in the same directory as the `.md` file
