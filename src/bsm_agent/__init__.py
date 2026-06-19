"""Pure Python tools for SM-gauge renormalizable model building."""

from .fields import Field, FieldFactor, FieldKind
from .ewsb import EWSBResult, GenericEWSBResult, VEVExpansion, expand_around_vevs, sm_ewsb
from .interactions import (
    GaugeInteractionTerm,
    GaugeSelfInteractionTerm,
    SeagullTerm,
    gauge_interaction_terms,
    gauge_interactions_latex,
    gauge_self_interaction_terms,
    gauge_self_interactions_latex,
    scalar_seagull_latex,
    scalar_seagull_terms,
)
from .indexed import indexed_operator_latex
from .latex import latex_document, write_latex, write_pdf
from .mass_matrices import MassMatrixBlock, MassMatrixResult, component_label, compute_mass_matrices, neutral_scalar_vev_shifts, neutral_scalar_vev_substitutions
from .model import Model
from .operators import Lagrangian, Operator
from .remote_api import (
    AnthropicChatModel,
    ModelTarget,
    OpenAICompatibleChatModel,
    OpenWebUIChatModel,
    build_chat_model,
    format_model_error,
    normalize_model_alias,
    normalize_openai_base_url,
    pick_default_model,
    resolve_anthropic_settings,
    resolve_model_target,
    resolve_openai_settings,
    resolve_remote_settings,
)
from .sm import StandardModel

__all__ = [
    "Field",
    "FieldFactor",
    "FieldKind",
    "AnthropicChatModel",
    "EWSBResult",
    "GenericEWSBResult",
    "GaugeInteractionTerm",
    "GaugeSelfInteractionTerm",
    "Lagrangian",
    "MassMatrixBlock",
    "MassMatrixResult",
    "Model",
    "Operator",
    "OpenAICompatibleChatModel",
    "OpenWebUIChatModel",
    "SeagullTerm",
    "StandardModel",
    "VEVExpansion",
    "ModelTarget",
    "build_chat_model",
    "component_label",
    "compute_mass_matrices",
    "expand_around_vevs",
    "format_model_error",
    "latex_document",
    "gauge_interaction_terms",
    "gauge_interactions_latex",
    "gauge_self_interaction_terms",
    "gauge_self_interactions_latex",
    "indexed_operator_latex",
    "normalize_model_alias",
    "normalize_openai_base_url",
    "neutral_scalar_vev_shifts",
    "neutral_scalar_vev_substitutions",
    "pick_default_model",
    "resolve_anthropic_settings",
    "resolve_model_target",
    "resolve_openai_settings",
    "resolve_remote_settings",
    "write_latex",
    "write_pdf",
    "scalar_seagull_latex",
    "scalar_seagull_terms",
    "sm_ewsb",
]
