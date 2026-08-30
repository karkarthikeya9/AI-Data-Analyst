from pandasai import Agent


def create_data_agent(df):
    """Create a PandasAI agent for the provided DataFrame."""
    return Agent(df)


def create_analyze_data_tool(data_agent):
    """Create the analyze_data tool bound to a PandasAI agent."""

    def analyze_data(question):
        """Analyze the dataset using PandasAI."""
        result = data_agent.chat(question)
        return str(result)

    return analyze_data