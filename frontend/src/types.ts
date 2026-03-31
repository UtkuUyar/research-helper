export type Message = { 
    role: 'user' | 'assistant'; 
    content: string
};

export type Session = { 
    id: string; 
    title: string; 
    messages: Message[] 
};

export type PaperSummary = {
    research_problem: string;
    key_contributions: string[];
    method_overview: string;
    experimental_findings: string[];
    limitations: string[];
};