# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Multi-agent financial research system with Analyst, Auditor, and Retriever agents
- FastAPI REST API for research operations
- SEC API integration for real-time financial filings
- Vector-based semantic document retrieval with Qdrant
- LangGraph workflow orchestration for agent coordination
- Human-in-the-loop approval checkpoints
- RAGAS evaluation framework for quality assurance
- Financial calculation tools for deterministic analysis
- Compliance reporting system
- Web UI for research workspace, landing page, and authentication
- Docker and Docker Compose support for containerized deployment
- Comprehensive test suite with pytest
- LangSmith integration for LLM observability

### Changed
- Modernized README with badges, emojis, and structured sections
- Enhanced project documentation with architecture diagrams
- Improved API documentation with example requests/responses

### Fixed
- Resolved timeout issues in research endpoints
- Fixed edge cases in compliance verification
- Improved error handling in vector store operations

## [1.0.0] - 2026-09-01

### Added
- Initial public release of AUGUS
- Financial Agentic RAG system with multi-agent architecture
- Production-oriented implementation for SEC financial research
- Autonomous agent workflow with human approval gates
- Deterministic financial calculations
- Compliance routing system
- Provenance-preserving retrieval pipeline
- Full test coverage with unit and integration tests
- Docker containerization support
- Static web UI with landing page, login, and workspace

## Standards

### Version Numbering
- MAJOR version for incompatible API changes
- MINOR version for added functionality in backward-compatible manner
- PATCH version for backward-compatible bug fixes

### Release Process
1. Update version in relevant files
2. Update CHANGELOG.md with changes
3. Create git tag: `git tag -a v1.0.0 -m "Release version 1.0.0"`
4. Push tag: `git push origin v1.0.0`
5. Create GitHub Release with changelog

## Future Roadmap

### Q4 2026
- [ ] WebSocket support for real-time research updates
- [ ] Advanced caching layer with Redis
- [ ] Multi-language support for international research
- [ ] Research history and audit trail

### Q1 2027
- [ ] Custom model fine-tuning capabilities
- [ ] Collaborative team workflows
- [ ] Export formats (PDF, Excel, JSON)
- [ ] Scheduled/automated research

### Q2 2027
- [ ] Browser extension for quick research
- [ ] Mobile app support
- [ ] Advanced analytics dashboard
- [ ] Custom domain integrations

---

For more information, visit [AUGUS Documentation](https://docs.augus.dev)
