module kinetics
    implicit none
contains
    ! Calculates the derivative of concentrations (dcdt) given reactant and product stoichiometries.
    subroutine calc_dcdt(concentrations, rate_constants, stoich_matrix_from, stoich_matrix_to, dcdt, n)
        implicit none
        integer, intent(in) :: n
        double precision, dimension(:), intent(in) :: concentrations, rate_constants
        integer, dimension(:,:), intent(in) :: stoich_matrix_from, stoich_matrix_to
        double precision, dimension(:), intent(inout) :: dcdt
        integer :: i, j
        double precision :: rate

        
        do i = 1, size(rate_constants)
            rate = rate_constants(i)
            
            ! Compute the contribution from reactants.
            do j = 1, n
                if (stoich_matrix_from(j, i) > 0) then
                    rate = rate * (concentrations(j) ** stoich_matrix_from(j, i))
                end if
            end do
            
            ! Update concentrations for reactants and products.
            do j = 1, n
                ! Subtract for reactants.
                if (stoich_matrix_from(j, i) > 0) then
                    dcdt(j) = dcdt(j) - rate * stoich_matrix_from(j, i)
                end if
                ! Add for products.
                if (stoich_matrix_to(j, i) > 0) then
                    dcdt(j) = dcdt(j) + rate * stoich_matrix_to(j, i)
                end if
            end do
        end do
    end subroutine calc_dcdt
end module kinetics

