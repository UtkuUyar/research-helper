export type Message = { 
    role: 'user' | 'assistant'; 
    content: string
};

export type Session = { 
    id: string; 
    title: string;
    created_at: number;
    session_ttl: number;
    messages: Message[] 
};

export type PaperSummary = {
    research_problem: string;
    key_contributions: string[];
    method_overview: string;
    experimental_findings: string[];
    limitations: string[];
};