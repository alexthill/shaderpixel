use std::ops;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Rad<T>(pub T);

impl From<Deg<f32>> for Rad<f32> {
    fn from(value: Deg<f32>) -> Self {
        Rad(value.0 / 180. * std::f32::consts::PI)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Deg<T>(pub T);

impl From<Rad<f32>> for Deg<f32> {
    fn from(value: Rad<f32>) -> Self {
        Deg(value.0 * 180. / std::f32::consts::PI)
    }
}

impl<T: Default> Default for Deg<T> {
    fn default() -> Self {
        Deg(T::default())
    }
}

impl<T: ops::Add> ops::Add for Deg<T> {
    type Output = Deg<T::Output>;

    fn add(self, rhs: Self) -> Self::Output {
        Deg(self.0 + rhs.0)
    }
}

impl<T: ops::AddAssign> ops::AddAssign for Deg<T> {
    fn add_assign(&mut self, rhs: Self) {
       self.0 += rhs.0
    }
}

impl<T: ops::Mul> ops::Mul for Deg<T> {
    type Output = Deg<T::Output>;

    fn mul(self, rhs: Self) -> Self::Output {
        Deg(self.0 * rhs.0)
    }
}

impl<T: ops::Div> ops::Div for Deg<T> {
    type Output = Deg<T::Output>;

    fn div(self, rhs: Self) -> Self::Output {
        Deg(self.0 / rhs.0)
    }
}
