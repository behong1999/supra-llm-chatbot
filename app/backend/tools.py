from langchain_core.tools import BaseTool
from langchain_community.tools import DuckDuckGoSearchRun
from duckduckgo_search.exceptions import TimeoutException


class HoasApartment(BaseTool):
    """
    This is a tool for searching HOAS apartments.
    """
    name = "Apartment assistant"
    description = "Use this tool when you need to search for apartments in Finland on the HOAS website."

    def _run(self, query: str):
        try:
            docs = DuckDuckGoSearchRun().run(f"site:hoas.fi/en/ {query}") + "\n"
        except TimeoutException as ex:
            docs = str(ex) + "\n"
        except Exception as e:
            docs = f"An error occurred: {e}" + "\n"
        return docs

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")


class ImmigrationRP(BaseTool):
    """
    This is a tool for searching migri.fi for immigration related topics.
    """
    name = "Resident permit assistant"
    description = "Use this tool when you need to search for information about applying for a Finnish resident permit on the Migri website."

    def _run(self, query: str):
        try:
            docs = DuckDuckGoSearchRun().run(f"site:hoas.fi/en/ {query}") + "\n"
        except TimeoutException as ex:
            docs = str(ex) + "\n"
        except Exception as e:
            docs = f"An error occurred: {e}" + "\n"
        return docs

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")


class StudyProgramme(BaseTool):
    """
    This is a tool for searching opintopolku.fi for information about study programmes in Finland.
    """
    name = "Study programme selection assistant"
    description = "Use this tool when you need to search for information about study programmes in Finland on the Opintopolku website."

    def _run(self, query: str):
        try:
            docs = DuckDuckGoSearchRun().run(f"site:opintopolku.fi/konfo/en/ {query}") + "\n"
        except TimeoutException as ex:
            docs = str(ex) + "\n"
        except Exception as e:
            docs = f"An error occurred: {e}" + "\n"
        return docs

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")