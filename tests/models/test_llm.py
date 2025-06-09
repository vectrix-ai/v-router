import pytest
from pydantic import ValidationError

from v_router.classes.llm import LLM, BackupModel


class TestBackupModel:
    """Test cases for the BackupModel class."""
    
    def test_backup_model_creation_valid(self):
        """Test creating a valid BackupModel."""
        primary_model = LLM(
            model_name="gpt-4.1-nano",
            provider="openai"
        )
        
        backup = BackupModel(
            model=primary_model,
            priority=1
        )
        
        assert backup.model.model_name == "gpt-4.1-nano"
        assert backup.model.provider == "openai"
        assert backup.priority == 1
    
    def test_backup_model_priority_validation_valid(self):
        """Test that valid priorities are accepted."""
        primary_model = LLM(
            model_name="gpt-4.1-nano",
            provider="openai"
        )
        
        # Test various valid priorities
        for priority in [1, 2, 5, 10, 100]:
            backup = BackupModel(
                model=primary_model,
                priority=priority
            )
            assert backup.priority == priority
    
    def test_backup_model_priority_validation_invalid(self):
        """Test that invalid priorities are rejected."""
        primary_model = LLM(
            model_name="gpt-4.1-nano",
            provider="openai"
        )
        
        # Test invalid priorities (less than 1)
        for invalid_priority in [0, -1, -5]:
            with pytest.raises(ValidationError) as exc_info:
                BackupModel(
                    model=primary_model,
                    priority=invalid_priority
                )
            assert "Priority must be at least 1" in str(exc_info.value)


class TestLLM:
    """Test cases for the LLM class."""
    
    def test_llm_creation_minimal(self):
        """Test creating an LLM with minimal required fields."""
        llm = LLM(
            model_name="gpt-4.1-nano",
            provider="openai"
        )
        
        assert llm.model_name == "gpt-4.1-nano"
        assert llm.provider == "openai"
        assert llm.max_tokens is None
        assert llm.temperature == 0  # Default value
        assert llm.backup_models == []  # Default empty list
    
    def test_llm_creation_full(self):
        """Test creating an LLM with all fields specified."""
        llm = LLM(
            model_name="claude-sonnet-3.5",
            provider="anthropic",
            max_tokens=4000,
            temperature=0.8,
            backup_models=[]
        )
        
        assert llm.model_name == "claude-sonnet-3.5"
        assert llm.provider == "anthropic"
        assert llm.max_tokens == 4000
        assert llm.temperature == 0.8
        assert llm.backup_models == []
    
    def test_llm_provider_validation(self):
        """Test that only valid providers are accepted."""
        valid_providers = ["openai", "anthropic", "azure", "google", "vertexai"]
        
        # Test valid providers
        for provider in valid_providers:
            llm = LLM(
                model_name="test-model",
                provider=provider
            )
            assert llm.provider == provider
        
        # Test invalid provider
        with pytest.raises(ValidationError):
            LLM(
                model_name="test-model",
                provider="invalid_provider"
            )
    
    def test_llm_with_single_backup(self):
        """Test LLM with a single backup model."""
        backup_model = LLM(
            model_name="gpt-4.1-nano",
            provider="openai"
        )
        
        backup = BackupModel(
            model=backup_model,
            priority=1
        )
        
        primary = LLM(
            model_name="gpt-4.1-nano",
            provider="openai",
            backup_models=[backup]
        )
        
        assert len(primary.backup_models) == 1
        assert primary.backup_models[0].model.model_name == "gpt-4.1-nano"
        assert primary.backup_models[0].priority == 1
    
    def test_llm_with_multiple_backups(self):
        """Test LLM with multiple backup models."""
        backup1 = BackupModel(
            model=LLM(model_name="claude-sonnet-3.5", provider="anthropic"),
            priority=1
        )
        backup2 = BackupModel(
            model=LLM(model_name="gemini-pro", provider="google"),
            priority=2
        )
        backup3 = BackupModel(
            model=LLM(model_name="gpt-4.1-nano", provider="openai"),
            priority=3
        )
        
        primary = LLM(
            model_name="gpt-4.1-nano",
            provider="openai",
            backup_models=[backup1, backup2, backup3]
        )
        
        assert len(primary.backup_models) == 3
        assert primary.backup_models[0].priority == 1
        assert primary.backup_models[1].priority == 2
        assert primary.backup_models[2].priority == 3
    
    def test_llm_backup_models_unique_priorities_valid(self):
        """Test that backup models with unique priorities are accepted."""
        backup1 = BackupModel(
            model=LLM(model_name="claude-sonnet-3.5", provider="anthropic"),
            priority=1
        )
        backup2 = BackupModel(
            model=LLM(model_name="gemini-pro", provider="google"),
            priority=3
        )
        backup3 = BackupModel(
            model=LLM(model_name="gpt-4.1-nano", provider="openai"),
            priority=5
        )
        
        # Should not raise an exception
        llm = LLM(
            model_name="gpt-4.1-nano",
            provider="openai",
            backup_models=[backup1, backup2, backup3]
        )
        
        assert len(llm.backup_models) == 3
    
    def test_llm_backup_models_duplicate_priorities_invalid(self):
        """Test that backup models with duplicate priorities are rejected."""
        backup1 = BackupModel(
            model=LLM(model_name="claude-sonnet-3.5", provider="anthropic"),
            priority=1
        )
        backup2 = BackupModel(
            model=LLM(model_name="gemini-pro", provider="google"),
            priority=1  # Duplicate priority
        )
        
        with pytest.raises(ValidationError) as exc_info:
            LLM(
                model_name="gpt-4.1-nano",
                provider="openai",
                backup_models=[backup1, backup2]
            )
        
        assert "Backup models must have unique priorities" in str(exc_info.value)
    
    def test_llm_backup_models_multiple_duplicates_invalid(self):
        """Test that multiple duplicate priorities are rejected."""
        backup1 = BackupModel(
            model=LLM(model_name="claude-sonnet-3.5", provider="anthropic"),
            priority=2
        )
        backup2 = BackupModel(
            model=LLM(model_name="gemini-pro", provider="google"),
            priority=1
        )
        backup3 = BackupModel(
            model=LLM(model_name="gpt-4.1-nano", provider="openai"),
            priority=2  # Duplicate of backup1
        )
        
        with pytest.raises(ValidationError) as exc_info:
            LLM(
                model_name="gpt-4.1-nano",
                provider="openai",
                backup_models=[backup1, backup2, backup3]
            )
        
        assert "Backup models must have unique priorities" in str(exc_info.value)
    
    def test_get_ordered_backup_models_empty(self):
        """Test get_ordered_backup_models with no backup models."""
        llm = LLM(
            model_name="gpt-4.1-nano",
            provider="openai"
        )
        
        ordered_backups = llm.get_ordered_backup_models()
        assert ordered_backups == []
        assert isinstance(ordered_backups, list)
    
    def test_get_ordered_backup_models_single(self):
        """Test get_ordered_backup_models with a single backup model."""
        backup = BackupModel(
            model=LLM(model_name="claude-sonnet-3.5", provider="anthropic"),
            priority=1
        )
        
        llm = LLM(
            model_name="gpt-4.1-nano",
            provider="openai",
            backup_models=[backup]
        )
        
        ordered_backups = llm.get_ordered_backup_models()
        assert len(ordered_backups) == 1
        assert ordered_backups[0].model_name == "claude-sonnet-3.5"
        assert ordered_backups[0].provider == "anthropic"
    
    def test_get_ordered_backup_models_multiple_ordered(self):
        """Test get_ordered_backup_models with multiple models in priority order."""
        backup1 = BackupModel(
            model=LLM(model_name="claude-sonnet-3.5", provider="anthropic"),
            priority=1
        )
        backup2 = BackupModel(
            model=LLM(model_name="gemini-pro", provider="google"),
            priority=2
        )
        backup3 = BackupModel(
            model=LLM(model_name="gpt-4.1-nano", provider="openai"),
            priority=3
        )
        
        llm = LLM(
            model_name="gpt-4.1-nano",
            provider="openai",
            backup_models=[backup1, backup2, backup3]
        )
        
        ordered_backups = llm.get_ordered_backup_models()
        assert len(ordered_backups) == 3
        assert ordered_backups[0].model_name == "claude-sonnet-3.5"  # Priority 1
        assert ordered_backups[1].model_name == "gemini-pro"         # Priority 2
        assert ordered_backups[2].model_name == "gpt-4.1-nano"     # Priority 3
    
    def test_get_ordered_backup_models_multiple_unordered(self):
        """Test get_ordered_backup_models with models added out of priority order."""
        backup1 = BackupModel(
            model=LLM(model_name="claude-sonnet-3.5", provider="anthropic"),
            priority=3  # Highest number, lowest priority
        )
        backup2 = BackupModel(
            model=LLM(model_name="gemini-pro", provider="google"),
            priority=1  # Lowest number, highest priority
        )
        backup3 = BackupModel(
            model=LLM(model_name="gpt-4.1-nano", provider="openai"),
            priority=2
        )
        
        # Add them in non-priority order
        llm = LLM(
            model_name="gpt-4.1-nano",
            provider="openai",
            backup_models=[backup1, backup2, backup3]
        )
        
        ordered_backups = llm.get_ordered_backup_models()
        assert len(ordered_backups) == 3
        assert ordered_backups[0].model_name == "gemini-pro"         # Priority 1
        assert ordered_backups[1].model_name == "gpt-4.1-nano"     # Priority 2
        assert ordered_backups[2].model_name == "claude-sonnet-3.5" # Priority 3
    
    def test_get_ordered_backup_models_large_priority_gaps(self):
        """Test get_ordered_backup_models with large gaps in priority numbers."""
        backup1 = BackupModel(
            model=LLM(model_name="claude-sonnet-3.5", provider="anthropic"),
            priority=100
        )
        backup2 = BackupModel(
            model=LLM(model_name="gemini-pro", provider="google"),
            priority=5
        )
        backup3 = BackupModel(
            model=LLM(model_name="gpt-4.1-nano", provider="openai"),
            priority=1
        )
        
        llm = LLM(
            model_name="gpt-4.1-nano",
            provider="openai",
            backup_models=[backup1, backup2, backup3]
        )
        
        ordered_backups = llm.get_ordered_backup_models()
        assert len(ordered_backups) == 3
        assert ordered_backups[0].model_name == "gpt-4.1-nano"     # Priority 1
        assert ordered_backups[1].model_name == "gemini-pro"        # Priority 5
        assert ordered_backups[2].model_name == "claude-sonnet-3.5" # Priority 100


class TestLLMSelfReference:
    """Test cases for LLM self-reference (backup models having their own backups)."""
    
    def test_llm_nested_backup_models(self):
        """Test LLM with backup models that themselves have backup models."""
        # Create a deep backup model
        deep_backup = LLM(
            model_name="gpt-4.1-nano",
            provider="openai"
        )
        
        # Create a mid-level backup with its own backup
        mid_backup = LLM(
            model_name="gemini-pro",
            provider="google",
            backup_models=[
                BackupModel(model=deep_backup, priority=1)
            ]
        )
        
        # Create a primary model with nested backups
        primary = LLM(
            model_name="gpt-4.1-nano",
            provider="openai",
            backup_models=[
                BackupModel(model=mid_backup, priority=1)
            ]
        )
        
        # Verify the structure
        assert primary.model_name == "gpt-4.1-nano"
        assert len(primary.backup_models) == 1
        
        first_backup = primary.backup_models[0].model
        assert first_backup.model_name == "gemini-pro"
        assert len(first_backup.backup_models) == 1
        
        second_backup = first_backup.backup_models[0].model
        assert second_backup.model_name == "gpt-4.1-nano"
        assert len(second_backup.backup_models) == 0
    
    def test_llm_complex_nested_structure(self):
        """Test a complex nested backup structure with multiple levels."""
        # Level 3 models (deepest)
        level3_model1 = LLM(model_name="gpt-4.1-nano", provider="openai")
        level3_model2 = LLM(model_name="gemini-pro", provider="google")
        
        # Level 2 models with their own backups
        level2_model1 = LLM(
            model_name="claude-sonnet-3.5",
            provider="anthropic",
            backup_models=[
                BackupModel(model=level3_model1, priority=1),
                BackupModel(model=level3_model2, priority=2)
            ]
        )
        
        level2_model2 = LLM(
            model_name="gemini-pro-1.5",
            provider="google",
            backup_models=[
                BackupModel(model=level3_model1, priority=1)
            ]
        )
        
        # Level 1 (primary) model
        primary = LLM(
            model_name="gpt-4.1-nano",
            provider="openai",
            backup_models=[
                BackupModel(model=level2_model1, priority=1),
                BackupModel(model=level2_model2, priority=2)
            ]
        )
        
        # Verify the complex structure
        assert len(primary.backup_models) == 2
        
        # Check first backup path
        first_backup = primary.backup_models[0].model
        assert first_backup.model_name == "claude-sonnet-3.5"
        assert len(first_backup.backup_models) == 2
        
        # Check second backup path
        second_backup = primary.backup_models[1].model
        assert second_backup.model_name == "gemini-pro-1.5"
        assert len(second_backup.backup_models) == 1


class TestLLMEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_llm_missing_required_fields(self):
        """Test that missing required fields raise ValidationError."""
        # Missing model_name
        with pytest.raises(ValidationError):
            LLM(provider="openai")
        
        # Missing provider
        with pytest.raises(ValidationError):
            LLM(model_name="gpt-4.1-nano")
        
        # Missing both
        with pytest.raises(ValidationError):
            LLM()
    
    def test_llm_temperature_boundaries(self):
        """Test temperature field with boundary values."""
        # Valid temperatures
        for temp in [0.0, 0.5, 1.0, 1.5, 2.0]:
            llm = LLM(
                model_name="gpt-4.1-nano",
                provider="openai",
                temperature=temp
            )
            assert llm.temperature == temp
    
    def test_llm_max_tokens_values(self):
        """Test max_tokens field with various values."""
        # Valid max_tokens
        for tokens in [None, 1, 100, 4000, 8000]:
            llm = LLM(
                model_name="gpt-4.1-nano",
                provider="openai",
                max_tokens=tokens
            )
            assert llm.max_tokens == tokens
    
    def test_backup_model_missing_required_fields(self):
        """Test that BackupModel requires both model and priority."""
        primary_model = LLM(model_name="gpt-4.1-nano", provider="openai")
        
        # Missing priority
        with pytest.raises(ValidationError):
            BackupModel(model=primary_model)
        
        # Missing model
        with pytest.raises(ValidationError):
            BackupModel(priority=1)
        
        # Missing both
        with pytest.raises(ValidationError):
            BackupModel()


class TestLLMUsagePatterns:
    """Test common usage patterns and integration scenarios."""
    
    def test_realistic_fallback_chain(self):
        """Test a realistic fallback chain for production use."""
        # Create a realistic fallback configuration
        primary = LLM(
            model_name="gpt-4.1-nano",
            provider="openai",
            max_tokens=4000,
            temperature=0.7,
            backup_models=[
                BackupModel(
                    model=LLM(
                        model_name="claude-sonnet-3.5",
                        provider="anthropic",
                        max_tokens=4000,
                        temperature=0.7
                    ),
                    priority=1
                ),
                BackupModel(
                    model=LLM(
                        model_name="gemini-pro",
                        provider="google",
                        max_tokens=4000,
                        temperature=0.7
                    ),
                    priority=2
                ),
                BackupModel(
                    model=LLM(
                        model_name="gpt-4.1-nano",
                        provider="openai",
                        max_tokens=4000,
                        temperature=0.7
                    ),
                    priority=3
                )
            ]
        )
        
        # Test the fallback ordering
        ordered_backups = primary.get_ordered_backup_models()
        expected_models = ["claude-sonnet-3.5", "gemini-pro", "gpt-4.1-nano"]
        actual_models = [backup.model_name for backup in ordered_backups]
        
        assert actual_models == expected_models
        
        # Verify all models have consistent configuration
        all_models = [primary] + ordered_backups
        for model in all_models:
            assert model.max_tokens == 4000
            assert model.temperature == 0.7
    
    def test_json_serialization_compatibility(self):
        """Test that the models can be serialized and deserialized."""
        backup = BackupModel(
            model=LLM(
                model_name="claude-sonnet-3.5",
                provider="anthropic"
            ),
            priority=1
        )
        
        primary = LLM(
            model_name="gpt-4.1-nano",
            provider="openai",
            backup_models=[backup]
        )
        
        # Test serialization
        json_data = primary.model_dump()
        assert isinstance(json_data, dict)
        assert json_data["model_name"] == "gpt-4.1-nano"
        assert json_data["provider"] == "openai"
        assert len(json_data["backup_models"]) == 1
        
        # Test deserialization
        reconstructed = LLM.model_validate(json_data)
        assert reconstructed.model_name == primary.model_name
        assert reconstructed.provider == primary.provider
        assert len(reconstructed.backup_models) == len(primary.backup_models)
        assert reconstructed.backup_models[0].priority == backup.priority
