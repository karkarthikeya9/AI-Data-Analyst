def create_tools(data_agent):
    from tools.data_analysis import create_analyze_data_tool

    analyze_data = create_analyze_data_tool(data_agent)

    return {
        "analyze_data": analyze_data,
    }