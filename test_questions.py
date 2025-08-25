# Test dataset for RAG performance evaluation

test_cases = [
    # Alice in Wonderland questions
    {
        "question": "Who is Alice?",
        "expected_topics": ["alice", "girl", "character", "wonderland"],
        "expected_source": "alice_in_wonderland.md",
        "category": "character_identification"
    },
    {
        "question": "What does Alice see when she falls down the rabbit hole?",
        "expected_topics": ["rabbit", "hole", "falling", "curious"],
        "expected_source": "alice_in_wonderland.md", 
        "category": "plot_details"
    },
    {
        "question": "Who does Alice meet at the tea party?",
        "expected_topics": ["mad", "hatter", "tea", "party"],
        "expected_source": "alice_in_wonderland.md",
        "category": "character_interaction"
    },
    {
        "question": "What happens when Alice drinks from the bottle?",
        "expected_topics": ["drink", "shrink", "grow", "size", "bottle"],
        "expected_source": "alice_in_wonderland.md",
        "category": "plot_details"
    },
    
    # Transformer/Attention paper questions
    {
        "question": "What is the Transformer architecture?",
        "expected_topics": ["transformer", "architecture", "attention", "encoder", "decoder"],
        "expected_source": "NIPS-2017-attention-is-all-you-need-Paper.pdf",
        "category": "architecture_overview"
    },
    {
        "question": "How does multi-head attention work?",
        "expected_topics": ["multi-head", "attention", "parallel", "heads", "linear"],
        "expected_source": "NIPS-2017-attention-is-all-you-need-Paper.pdf",
        "category": "technical_mechanism"
    },
    {
        "question": "What is scaled dot-product attention?",
        "expected_topics": ["scaled", "dot-product", "attention", "softmax", "queries", "keys"],
        "expected_source": "NIPS-2017-attention-is-all-you-need-Paper.pdf",
        "category": "technical_mechanism"
    },
    {
        "question": "Why do Transformers not use recurrence or convolution?",
        "expected_topics": ["recurrence", "convolution", "parallelization", "sequential", "attention"],
        "expected_source": "NIPS-2017-attention-is-all-you-need-Paper.pdf",
        "category": "design_rationale"
    },
    {
        "question": "What are the advantages of self-attention over recurrent layers?",
        "expected_topics": ["self-attention", "recurrent", "parallel", "path", "computation"],
        "expected_source": "NIPS-2017-attention-is-all-you-need-Paper.pdf",
        "category": "comparison"
    },
    {
        "question": "How do positional encodings work in Transformers?",
        "expected_topics": ["positional", "encoding", "sine", "cosine", "position"],
        "expected_source": "NIPS-2017-attention-is-all-you-need-Paper.pdf",
        "category": "technical_mechanism"
    },
    
    # Cross-document questions (should fail or return mixed results)
    {
        "question": "How does Alice use attention mechanisms?",
        "expected_topics": [],  # This should not have good results
        "expected_source": "none",
        "category": "cross_domain_invalid"
    },
    {
        "question": "What is the BLEU score for Alice in Wonderland?",
        "expected_topics": [],  # This should not have good results  
        "expected_source": "none",
        "category": "cross_domain_invalid"
    }
]