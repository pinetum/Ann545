.PHONY: clean All

All:
	@echo "----------Building project:[ RBF - Debug ]----------"
	@cd "RBF" && "$(MAKE)" -f  "RBF.mk"
clean:
	@echo "----------Cleaning project:[ RBF - Debug ]----------"
	@cd "RBF" && "$(MAKE)" -f  "RBF.mk" clean
