"""
Returns a hand-crafted Mermaid diagram for the agentic workflow.

Shows every agent/component including SupervisorAgent, SQLGenAgent, RAGEngine,
VectorStoreManager, and the new "both" hybrid route.
"""

_DIAGRAM = """\
%%{init: {'flowchart': {'curve': 'linear'}}}%%
flowchart TD
    START(["__start__"])
    END1(["__end__"])
    END2(["__end__"])

    START --> SUPERVISOR["SupervisorAgent<br/><i>(facade / entry point)</i>"]
    SUPERVISOR --> CHITCHAT["chitchat_detector"]

    CHITCHAT -->|chitchat / greeting| END1
    CHITCHAT -->|continue| ROUTER["RouterAgent"]

    ROUTER -->|table| TABLE_AGENT
    ROUTER -->|both| TABLE_AGENT
    ROUTER -->|doc| DOC_AGENT

    subgraph TABLE_AGENT["table_agent"]
        TA["TableAgent"] --> SGA["SQLGenAgent<br/><i>(LLM → SQL)</i>"]
    end

    TABLE_AGENT -->|"answer found<br/>(table or both routes)"| DOC_AGENT
    TABLE_AGENT -->|"no answer<br/>(table route)"| DOC_AGENT

    subgraph DOC_AGENT["doc_image_agent"]
        DIA["DocImageAgent"] --> RAGE["RAGEngine<br/><i>(LLM + prompt)</i>"]
        RAGE --> VSM["VectorStoreManager<br/><i>(ChromaDB)</i>"]
    end

    DOC_AGENT --> GRADER["GradingAgent"]

    GRADER --> HALL["HallucinationAgent"]
    HALL --> END2

    classDef default fill:#3b82f6,color:#000000,stroke:#1d4ed8,stroke-width:1px
    classDef terminal fill:#e0e7ff,color:#000000,stroke:#6366f1,stroke-width:1px
    classDef subcomp fill:#60a5fa,color:#000000,stroke:#2563eb,stroke-width:1px

    class START,END1,END2 terminal
    class SUPERVISOR,CHITCHAT,ROUTER,GRADER,HALL default
    class TA,SGA,DIA,RAGE,VSM subcomp
"""


def get_workflow_mermaid() -> str:
    return _DIAGRAM
