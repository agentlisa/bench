# LISA-Bench: Smart Contract Vulnerability Detection Benchmark
LISA-Bench: Advancing smart contract security through comprehensive, production-grade vulnerability data.

## Our Benchmark
Our dataset consists of 10,185 verified smart contract vulnerabilities with complete code context, derived from 10 authoritative security audit platforms including Code4rena, OpenZeppelin, Halborn, Sherlock, and TrailOfBits. This represents the most comprehensive collection of professionally-audited blockchain security issues spanning 2016-2024.
The benchmark covers 584 distinct protocols across DeFi, Layer 2, NFT, governance, and infrastructure projects. Each vulnerability includes complete code snippets, expert analysis from 3,086 security specialists, impact assessments, and remediation guidance. Every entry underwent professional security review, achieving >99.9% completeness across critical fields.

### Dataset Composition
Vulnerabilities across four severity categories:
- High Severity (1,685 cases, 16.5%): Critical issues leading to fund loss or system compromise
- Medium Severity (3,980 cases, 39.1%): Significant functional or security impairments
- Low Severity (3,217 cases, 31.6%): Code quality and best practice violations
- Gas Optimization (1,303 cases, 12.8%): Performance and cost inefficiencies

### Source Distribution
Code4rena (38.1%), OpenZeppelin (11.0%), Halborn (9.2%), Sherlock (7.7%), TrailOfBits (6.7%), plus 14 additional authoritative sources.

## Evaluation Framework
LISA-Bench provides standardized evaluation for vulnerability detection systems. For each case, the framework:
- Presents vulnerable code with full contract context for complete implementation analysis.
- Provides rich metadata: contract source, blockchain platform, project category, temporal information, and related vulnerabilities.
- Validates detection accuracy against expert-verified ground truth:
- Vulnerability identification: Correct vulnerability type detection
- Location precision: Exact vulnerable code pinpointing
- Severity assessment: Risk rating alignment with expert classification
- Root cause analysis: Accurate vulnerability mechanism explanation

### Evaluation Metrics
- Detection Rate: Percentage of vulnerabilities correctly identified
- False Positive Rate: Frequency of incorrect flags
- Severity Accuracy: Predicted vs. actual risk level alignment
- Explanation Quality: Correctness of vulnerability descriptions
- Coverage: Performance across different vulnerability classes

### Baseline Comparisons
- 25x more complete vulnerability cases than comparable datasets
- 8+ years of security evolution vs. point-in-time snapshots
- Professional verification by expert auditors vs. synthetic examples
- 1,219 production protocols across all major blockchain platforms

## Use Cases
- **AI/ML Research**: Train detection models, benchmark explainable AI, develop transfer learning techniques
- **Security Tools**: Validate static analysis accuracy, train automated audit systems, develop risk scoring algorithms
- **Education**: Teach secure coding, create training programs, build interactive learning platforms

## Data Format
```json
{
  "vulnerability_id": "unique identifier",
  "code_snippet": "complete vulnerable code",
  "vulnerability_type": "classification",
  "severity": "High/Medium/Low/Gas",
  "description": "expert analysis",
  "impact": "consequences and exploits",
  "recommendation": "remediation guidance",
  "audit_source": "platform",
  "project": "protocol and category",
  "audit_date": "discovery timestamp"
}
```

## Getting Started
LISA-Bench is designed for immediate integration into research and development workflows:
- Download the dataset in standardized JSON/CSV formats
- Select evaluation subsets by severity, vulnerability type, or time period
- Run foundation models using our reference implementation
- Compare results against published benchmarks and industry standards
- Contribute findings back to the research and AgentLISA community


