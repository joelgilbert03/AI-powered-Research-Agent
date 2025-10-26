from crewai.tools import BaseTool

class WebScraperTool(BaseTool):
    name: str = "Web Scraper Tool"
    description: str = "A dummy tool that simulates scraping a website."

    def _run(self, url: str) -> str:
        """Simulates scraping a website and returns placeholder content."""
        return f"Placeholder content for {url}. This tool is not fully implemented yet."
